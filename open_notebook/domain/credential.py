"""
Credential domain model for storing individual provider credentials.

Each credential is a standalone record in the 'credential' table, replacing
the old ProviderConfig singleton. Credentials store API keys (encrypted at
rest) and provider-specific configuration fields.

Usage:
    cred = Credential(
        name="Production",
        provider="openai",
        modalities=["language", "embedding"],
        api_key=SecretStr("sk-..."),
    )
    await cred.save()
"""

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from loguru import logger
from pydantic import SecretStr

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.base import ObjectModel
from open_notebook.utils.encryption import decrypt_value, encrypt_value


class Credential(ObjectModel):
    """
    Individual credential record for an AI provider.

    Each record stores authentication and configuration for a single provider
    account. Models link to credentials via the credential field.
    """

    table_name: ClassVar[str] = "credential"
    nullable_fields: ClassVar[set[str]] = {
        "api_key",
        "auth_token",
        "base_url",
        "endpoint",
        "api_version",
        "endpoint_llm",
        "endpoint_embedding",
        "endpoint_stt",
        "endpoint_tts",
        "project",
        "location",
        "credentials_path",
    }

    name: str
    provider: str
    modalities: List[str] = []
    api_key: Optional[SecretStr] = None
    auth_token: Optional[SecretStr] = None
    base_url: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    endpoint_llm: Optional[str] = None
    endpoint_embedding: Optional[str] = None
    endpoint_stt: Optional[str] = None
    endpoint_tts: Optional[str] = None
    project: Optional[str] = None
    location: Optional[str] = None
    credentials_path: Optional[str] = None

    def to_esperanto_config(self) -> Dict[str, Any]:
        """
        Build config dict for AIFactory.create_*() calls.

        Returns a dict that can be passed as the 'config' parameter to
        Esperanto's AIFactory methods, overriding env var lookup.
        """
        config: Dict[str, Any] = {}
        if self.api_key:
            config["api_key"] = self.api_key.get_secret_value()
        if self.auth_token:
            config["auth_token"] = self.auth_token.get_secret_value()
        logger.info(
            f"to_esperanto_config for {self.provider}: "
            f"has_api_key={self.api_key is not None}, "
            f"has_auth_token={self.auth_token is not None}, "
            f"config_keys={list(config.keys())}"
        )
        if self.base_url:
            config["base_url"] = self.base_url
        if self.endpoint:
            config["endpoint"] = self.endpoint
        if self.api_version:
            config["api_version"] = self.api_version
        if self.endpoint_llm:
            config["endpoint_llm"] = self.endpoint_llm
        if self.endpoint_embedding:
            config["endpoint_embedding"] = self.endpoint_embedding
        if self.endpoint_stt:
            config["endpoint_stt"] = self.endpoint_stt
        if self.endpoint_tts:
            config["endpoint_tts"] = self.endpoint_tts
        if self.project:
            config["project"] = self.project
        if self.location:
            config["location"] = self.location
        if self.credentials_path:
            config["credentials_path"] = self.credentials_path
        return config

    @classmethod
    async def get_by_provider(cls, provider: str) -> List["Credential"]:
        """Get all credentials for a provider."""
        results = await repo_query(
            "SELECT * FROM credential WHERE string::lowercase(provider) = string::lowercase($provider) ORDER BY created ASC",
            {"provider": provider},
        )
        credentials = []
        for row in results:
            try:
                cred = cls._from_db_row(row)
                credentials.append(cred)
            except Exception as e:
                logger.warning(f"Skipping invalid credential: {e}")
        return credentials

    @classmethod
    async def get(cls, id: str) -> "Credential":
        """Override get() to handle api_key and auth_token decryption."""
        instance = await super().get(id)
        # Pydantic auto-wraps the raw DB string in SecretStr, so we need
        # to extract, decrypt, and re-wrap regardless of type.
        for field_name in ("api_key", "auth_token"):
            field_val = getattr(instance, field_name, None)
            if field_val:
                raw = (
                    field_val.get_secret_value()
                    if isinstance(field_val, SecretStr)
                    else field_val
                )
                decrypted = decrypt_value(raw)
                object.__setattr__(instance, field_name, SecretStr(decrypted))
        return instance

    @classmethod
    async def get_all(cls, order_by=None) -> List["Credential"]:
        """Override get_all() to handle api_key and auth_token decryption."""
        instances = await super().get_all(order_by=order_by)
        for instance in instances:
            for field_name in ("api_key", "auth_token"):
                field_val = getattr(instance, field_name, None)
                if field_val:
                    raw = (
                        field_val.get_secret_value()
                        if isinstance(field_val, SecretStr)
                        else field_val
                    )
                    decrypted = decrypt_value(raw)
                    object.__setattr__(instance, field_name, SecretStr(decrypted))
        return instances

    async def get_linked_models(self) -> list:
        """Get all models linked to this credential."""
        if not self.id:
            return []
        from open_notebook.ai.models import Model

        results = await repo_query(
            "SELECT * FROM model WHERE credential = $cred_id",
            {"cred_id": ensure_record_id(self.id)},
        )
        return [Model(**row) for row in results]

    def _prepare_save_data(self) -> Dict[str, Any]:
        """Override to encrypt api_key and auth_token before storage."""
        data = {}
        secret_fields = {"api_key", "auth_token"}
        dump = self.model_dump()
        logger.info(
            f"_prepare_save_data: model_dump keys={list(dump.keys())}, "
            f"auth_token in dump={'auth_token' in dump}, "
            f"auth_token attr={getattr(self, 'auth_token', 'MISSING')!r}"
        )
        for key, value in dump.items():
            if key in secret_fields:
                # Handle SecretStr: extract, encrypt, store
                field_val = getattr(self, key, None)
                if field_val:
                    secret_value = field_val.get_secret_value()
                    data[key] = encrypt_value(secret_value)
                else:
                    data[key] = None
            elif value is not None or key in self.__class__.nullable_fields:
                data[key] = value

        logger.info(f"_prepare_save_data: final data keys={list(data.keys())}")
        return data

    async def save(self) -> None:
        """Save credential, handling api_key/auth_token re-hydration after DB round-trip."""
        # Remember the original SecretStr values before save
        original_api_key = self.api_key
        original_auth_token = self.auth_token

        await super().save()

        # After save, secret fields may be set to encrypted strings
        # from the DB result. Restore the original SecretStr values.
        for field_name, original in [("api_key", original_api_key), ("auth_token", original_auth_token)]:
            if original:
                object.__setattr__(self, field_name, original)
            else:
                current = getattr(self, field_name, None)
                if current and isinstance(current, str):
                    decrypted = decrypt_value(current)
                    object.__setattr__(self, field_name, SecretStr(decrypted))

    @classmethod
    def _from_db_row(cls, row: dict) -> "Credential":
        """Create a Credential from a database row, decrypting secret fields."""
        for field_name in ("api_key", "auth_token"):
            val = row.get(field_name)
            if val and isinstance(val, str):
                decrypted = decrypt_value(val)
                row[field_name] = SecretStr(decrypted)
            elif val is None:
                row[field_name] = None
        return cls(**row)
