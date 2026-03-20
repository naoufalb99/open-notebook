"""
Patch Esperanto's AnthropicLanguageModel to support auth_token (Authorization: Bearer).

This allows using either x-api-key or Authorization: Bearer header for Anthropic API auth.
Run after `uv sync` to apply the patch to the installed Esperanto package.
"""

import glob
import importlib
import os
import re
import sys


def find_anthropic_module():
    """Find the Esperanto Anthropic provider file."""
    import esperanto.providers.llm.anthropic as mod
    return mod.__file__


def clear_pycache(filepath):
    """Remove stale .pyc files for the patched module."""
    directory = os.path.dirname(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    pycache_dir = os.path.join(directory, "__pycache__")
    if os.path.isdir(pycache_dir):
        for pyc in glob.glob(os.path.join(pycache_dir, f"{basename}.*.pyc")):
            os.remove(pyc)
            print(f"Removed stale bytecode: {pyc}")


def patch():
    filepath = find_anthropic_module()
    with open(filepath, "r") as f:
        content = f.read()

    # Skip if already patched
    if "_auth_token" in content:
        print(f"Already patched: {filepath}")
        return

    # 1. Patch __post_init__ to support auth_token
    content = content.replace(
        '''    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable."
            )

        # Set base URL
        self.base_url = self.base_url or "https://api.anthropic.com/v1"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()''',
        '''    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")

        # Extract auth_token from config if provided (Authorization: Bearer alternative)
        self._auth_token = self._config.get("auth_token")

        if not self.api_key and not self._auth_token:
            raise ValueError(
                "Anthropic API key or auth token not found. Set the ANTHROPIC_API_KEY "
                "environment variable, or pass api_key or auth_token in config."
            )

        # Set base URL
        self.base_url = self.base_url or "https://api.anthropic.com/v1"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()'''
    )

    # 2. Patch _get_headers to use Bearer when auth_token is set
    content = content.replace(
        '''    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Anthropic API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }''',
        '''    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Anthropic API requests.

        Uses Authorization: Bearer if auth_token is configured,
        otherwise uses x-api-key header.
        """
        headers: Dict[str, str] = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        else:
            headers["x-api-key"] = self.api_key
        return headers'''
    )

    # 3. Patch to_langchain to support auth_token
    # Find and replace the kwargs block in to_langchain
    old_langchain = '''        # Anthropic does not allow both temperature and top_p to be set
        # Prioritize temperature if both are provided
        kwargs = {
            "model": model_name,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        elif self.top_p is not None:
            kwargs["top_p"] = self.top_p

        return ChatAnthropic(**kwargs)'''

    new_langchain = '''        # Anthropic does not allow both temperature and top_p to be set
        # Prioritize temperature if both are provided
        kwargs = {
            "model": model_name,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key or "placeholder",
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        elif self.top_p is not None:
            kwargs["top_p"] = self.top_p

        lc_model = ChatAnthropic(**kwargs)

        # If using auth_token (Authorization: Bearer), replace the
        # underlying Anthropic clients with ones using auth_token.
        # ChatAnthropic doesn't natively support auth_token, so we
        # pre-populate its cached_property clients.
        if self._auth_token:
            import anthropic as _anthropic

            client_kwargs = {
                "auth_token": self._auth_token,
                "base_url": self.base_url,
            }
            lc_model.__dict__["_client"] = _anthropic.Anthropic(**client_kwargs)
            lc_model.__dict__["_async_client"] = _anthropic.AsyncAnthropic(
                **client_kwargs
            )

        return lc_model'''

    content = content.replace(old_langchain, new_langchain)

    with open(filepath, "w") as f:
        f.write(content)

    # Clear stale bytecode so Python uses the patched .py
    clear_pycache(filepath)

    print(f"Patched: {filepath}")


if __name__ == "__main__":
    patch()
