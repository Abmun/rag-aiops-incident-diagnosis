"""
src/ingestion/ticket_ingester.py
─────────────────────────────────
Ingestion connector for incident tickets.
Supports ServiceNow REST API, PagerDuty API,
and local JSON/CSV file ingestion for PoC/testing.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Iterator

import requests
import structlog

from .base import BaseIngester, DocumentType, OperationalDocument

logger = structlog.get_logger(__name__)


class LocalTicketIngester(BaseIngester):
    """
    Ingests incident tickets from local JSON files.
    Used for PoC testing and evaluation dataset loading.

    Expected JSON format (list of objects):
    [
      {
        "id": "INC001",
        "title": "Payment service timeout",
        "description": "...",
        "resolution_notes": "...",
        "priority": "P1",
        "service": "payments-service",
        "status": "resolved",
        "created_at": "2024-01-15T10:30:00Z",
        "resolved_at": "2024-01-15T11:45:00Z",
        "root_cause": "Database connection pool exhaustion",
        "tags": ["database", "performance"]
      }
    ]
    """

    def validate_connection(self) -> bool:
        path = self.config.get("path", "data/samples/incidents")
        return os.path.exists(path)

    def ingest(self) -> Iterator[OperationalDocument]:
        path = self.config.get("path", "data/samples/incidents")

        if os.path.isfile(path):
            files = [path]
        else:
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith((".json", ".csv"))
            ]

        for file_path in files:
            self.logger.info("Ingesting tickets file", path=file_path)
            yield from self._ingest_file(file_path)

    def _ingest_file(self, file_path: str) -> Iterator[OperationalDocument]:
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                tickets = json.load(f)
            for ticket in tickets:
                yield self._ticket_to_document(ticket)

        elif file_path.endswith(".csv"):
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield self._ticket_to_document(dict(row))

    def _ticket_to_document(self, ticket: dict) -> OperationalDocument:
        # Compose rich content text for embedding quality
        content_parts = []
        if ticket.get("title"):
            content_parts.append(f"Incident: {ticket['title']}")
        if ticket.get("priority"):
            content_parts.append(f"Priority: {ticket['priority']}")
        if ticket.get("service"):
            content_parts.append(f"Affected Service: {ticket['service']}")
        if ticket.get("description"):
            content_parts.append(f"Description:\n{ticket['description']}")
        if ticket.get("error_message"):
            content_parts.append(f"Error:\n{ticket['error_message']}")
        if ticket.get("resolution_notes"):
            content_parts.append(f"Resolution:\n{ticket['resolution_notes']}")
        if ticket.get("root_cause"):
            content_parts.append(f"Root Cause: {ticket['root_cause']}")

        content = self._clean_text("\n\n".join(content_parts))
        content = self._redact_sensitive(content)

        created = self._parse_datetime(ticket.get("created_at"))
        modified = self._parse_datetime(
            ticket.get("resolved_at") or ticket.get("created_at")
        )

        return OperationalDocument(
            document_id=f"ticket_{ticket.get('id', hash(content))}",
            source_system="ticket_system",
            document_type=DocumentType.INCIDENT_TICKET,
            title=ticket.get("title", "Untitled Incident"),
            content_text=content,
            created_at=created,
            last_modified=modified,
            metadata={
                "priority": ticket.get("priority", "unknown"),
                "service": ticket.get("service", "unknown"),
                "status": ticket.get("status", "unknown"),
                "root_cause": ticket.get("root_cause", ""),
                "tags": ticket.get("tags", []),
                "environment": ticket.get("environment", "production"),
            },
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)


class ServiceNowIngester(BaseIngester):
    """
    Ingests incident tickets from ServiceNow REST API.
    Requires ServiceNow instance URL, username, and password.
    """

    BASE_FIELDS = (
        "sys_id,number,short_description,description,close_notes,"
        "priority,state,u_service_name,sys_created_on,sys_updated_on,"
        "u_root_cause,u_resolution_summary,category"
    )

    def validate_connection(self) -> bool:
        url = self.config.get("instance_url")
        try:
            resp = requests.get(
                f"{url}/api/now/table/incident?sysparm_limit=1",
                auth=(self.config["username"], self.config["password"]),
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            self.logger.error("ServiceNow connection failed", error=str(e))
            return False

    def ingest(self) -> Iterator[OperationalDocument]:
        url = self.config["instance_url"]
        auth = (self.config["username"], self.config["password"])
        batch_size = self.config.get("batch_size", 100)
        lookback_days = self.config.get("lookback_days", 90)
        offset = 0

        while True:
            resp = requests.get(
                f"{url}/api/now/table/incident",
                auth=auth,
                params={
                    "sysparm_fields": self.BASE_FIELDS,
                    "sysparm_limit": batch_size,
                    "sysparm_offset": offset,
                    "sysparm_query": (
                        f"state=6^"  # Resolved
                        f"sys_created_on>javascript:gs.daysAgo({lookback_days})"
                    ),
                },
                timeout=30,
            )
            resp.raise_for_status()
            records = resp.json().get("result", [])
            if not records:
                break

            for record in records:
                yield self._record_to_document(record)

            offset += batch_size
            if len(records) < batch_size:
                break

    def _record_to_document(self, record: dict) -> OperationalDocument:
        content_parts = [
            f"Incident: {record.get('short_description', '')}",
            f"Category: {record.get('category', '')}",
            f"Priority: {record.get('priority', {}).get('display_value', '')}",
            f"Service: {record.get('u_service_name', '')}",
            f"Description:\n{record.get('description', '')}",
            f"Resolution:\n{record.get('close_notes', '')}",
            f"Root Cause: {record.get('u_root_cause', '')}",
        ]
        content = self._clean_text("\n\n".join(content_parts))
        content = self._redact_sensitive(content)

        def _dt(val):
            try:
                return datetime.strptime(val, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                return datetime.now(timezone.utc)

        return OperationalDocument(
            document_id=f"snow_{record.get('sys_id', '')}",
            source_system="servicenow",
            document_type=DocumentType.INCIDENT_TICKET,
            title=record.get("short_description", ""),
            content_text=content,
            created_at=_dt(record.get("sys_created_on", "")),
            last_modified=_dt(record.get("sys_updated_on", "")),
            metadata={
                "number": record.get("number", ""),
                "priority": record.get("priority", {}).get("display_value", ""),
                "service": record.get("u_service_name", ""),
                "state": record.get("state", {}).get("display_value", ""),
            },
        )
