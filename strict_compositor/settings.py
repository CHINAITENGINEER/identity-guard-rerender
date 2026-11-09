from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path
from threading import RLock
from typing import Any


PRIVATE_SETTINGS_DIR = Path.cwd().resolve() / "ui_workspace" / "private"
SETTINGS_PATH = PRIVATE_SETTINGS_DIR / "integrations.json"
SECRET_FIELDS = {"minimax_api_key", "runninghub_api_key"}
PUBLIC_DEFAULTS: dict[str, Any] = {
    "minimax_base_url": "https://api.minimax.io",
    "minimax_model": "MiniMax-M3",
    # ``auto`` keeps the normal Windows proxy behaviour first, then retries
    # the exact same approved endpoint without inherited proxy settings only
    # when the proxy has a transport failure.  It never changes provider host.
    "minimax_network_mode": "auto",
    "runninghub_base_url": "https://www.runninghub.cn",
    "runninghub_mode": "ai_app",
    # Public releases require an explicitly configured, audited workflow.
    "runninghub_cutout_id": "",
    "runninghub_input_node_id": "",
    "runninghub_input_field_name": "image",
    "runninghub_output_node_id": "",
    "runninghub_enabled": False,
    # Hybrid rendering is a separate image-edit workflow.  It must never
    # reuse the foreground-extraction app/node mapping above.
    "runninghub_hybrid_enabled": False,
    "runninghub_hybrid_mode": "ai_app",
    "runninghub_hybrid_app_id": "",
    "runninghub_hybrid_source_node_id": "",
    "runninghub_hybrid_source_field_name": "image",
    "runninghub_hybrid_mask_node_id": "",
    "runninghub_hybrid_mask_field_name": "mask",
    # Optional second mask input for inpaint/ComfyUI workflows that support a
    # protect/keep region in addition to the editable region.
    "runninghub_hybrid_protect_node_id": "",
    "runninghub_hybrid_protect_field_name": "mask",
    "runninghub_hybrid_prompt_node_id": "",
    "runninghub_hybrid_prompt_field_name": "prompt",
    "runninghub_hybrid_output_node_id": "",
    # Restricted learned-effects workflow.  Unlike hybrid rendering, this
    # workflow may return only shadow/gain/specular maps and can never return
    # a final composited RGB image to the core pipeline.
    "runninghub_learned_enabled": False,
    "runninghub_learned_mode": "comfyui",
    "runninghub_learned_app_id": "",
    "runninghub_learned_source_node_id": "",
    "runninghub_learned_source_field_name": "image",
    "runninghub_learned_product_mask_node_id": "",
    "runninghub_learned_product_mask_field_name": "image",
    "runninghub_learned_geometry_node_id": "",
    "runninghub_learned_geometry_field_name": "image",
    "runninghub_learned_normal_node_id": "",
    "runninghub_learned_normal_field_name": "image",
    "runninghub_learned_background_node_id": "",
    "runninghub_learned_background_field_name": "image",
    "runninghub_learned_prompt_node_id": "",
    "runninghub_learned_prompt_field_name": "prompt",
    "runninghub_learned_shadow_output_node_id": "",
    "runninghub_learned_gain_output_node_id": "",
    "runninghub_learned_specular_output_node_id": "",
    "runninghub_learned_shadow_encoding": "white_is_shadow",
}
_LOCK = RLock()
_ENTROPY = b"strict-compositor-integrations-v1"


class SettingsError(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob(data: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(data)
    return (
        _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))),
        buffer,
    )


def _protect_windows(value: str) -> str:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    source, source_buffer = _blob(value.encode("utf-8"))
    entropy, entropy_buffer = _blob(_ENTROPY)
    target = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        "Strict Compositor credential",
        ctypes.byref(entropy),
        None,
        None,
        0,
        ctypes.byref(target),
    ):
        raise ctypes.WinError()
    try:
        payload = ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)
    # Keep the input buffers alive until CryptProtectData returns.
    del source_buffer, entropy_buffer
    return "dpapi:" + base64.b64encode(payload).decode("ascii")


def _unprotect_windows(value: str) -> str:
    if not value.startswith("dpapi:"):
        raise SettingsError("Unsupported encrypted credential format")
    payload = base64.b64decode(value[6:].encode("ascii"), validate=True)
    source, source_buffer = _blob(payload)
    entropy, entropy_buffer = _blob(_ENTROPY)
    target = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        ctypes.byref(entropy),
        None,
        None,
        0,
        ctypes.byref(target),
    ):
        raise ctypes.WinError()
    try:
        clear = ctypes.string_at(target.pbData, target.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(target.pbData)
    del source_buffer, entropy_buffer
    return clear


def protect_secret(value: str) -> str:
    if os.name != "nt":
        raise SettingsError(
            "Persistent API-key storage is supported only through Windows DPAPI; "
            "use environment variables on this platform."
        )
    return _protect_windows(value)


def unprotect_secret(value: str) -> str:
    if os.name != "nt":
        raise SettingsError("Windows DPAPI is unavailable")
    return _unprotect_windows(value)


def _read() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingsError(f"Integration settings cannot be read: {exc}") from exc
    if not isinstance(value, dict):
        raise SettingsError("Integration settings must be a JSON object")
    return value


def _write(value: dict[str, Any]) -> None:
    PRIVATE_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, SETTINGS_PATH)


def get_secret(name: str) -> str:
    if name not in SECRET_FIELDS:
        raise KeyError(name)
    environment_name = name.upper()
    environment_value = os.environ.get(environment_name, "").strip()
    if environment_value:
        return environment_value
    with _LOCK:
        encrypted = _read().get("secrets", {}).get(name)
    if not encrypted:
        return ""
    try:
        return unprotect_secret(str(encrypted)).strip()
    except Exception as exc:
        raise SettingsError(f"Stored credential {name} cannot be decrypted") from exc


def get_public_settings() -> dict[str, Any]:
    with _LOCK:
        stored = _read()
    public = dict(PUBLIC_DEFAULTS)
    public.update({
        key: value
        for key, value in stored.get("public", {}).items()
        if key in PUBLIC_DEFAULTS
    })
    public["minimax_configured"] = bool(get_secret("minimax_api_key"))
    public["runninghub_configured"] = bool(get_secret("runninghub_api_key"))
    return public


def save_integrations(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_modes = {"ai_app", "comfyui"}
    allowed_minimax_network_modes = {"auto", "system_proxy", "direct"}
    public = {
        key: payload.get(key, default)
        for key, default in PUBLIC_DEFAULTS.items()
    }
    public["runninghub_mode"] = str(public["runninghub_mode"])
    public["runninghub_hybrid_mode"] = str(public["runninghub_hybrid_mode"])
    public["runninghub_learned_mode"] = str(public["runninghub_learned_mode"])
    if public["runninghub_mode"] not in allowed_modes:
        raise SettingsError("runninghub_mode must be ai_app or comfyui")
    if public["runninghub_hybrid_mode"] not in allowed_modes:
        raise SettingsError("runninghub_hybrid_mode must be ai_app or comfyui")
    if public["runninghub_learned_mode"] not in allowed_modes:
        raise SettingsError("runninghub_learned_mode must be ai_app or comfyui")
    allowed_shadow_encodings = {
        "white_is_shadow", "black_is_shadow", "alpha_channel",
    }
    public["runninghub_learned_shadow_encoding"] = str(
        public["runninghub_learned_shadow_encoding"]
    ).strip()
    if public["runninghub_learned_shadow_encoding"] not in allowed_shadow_encodings:
        raise SettingsError(
            "runninghub_learned_shadow_encoding must be white_is_shadow, "
            "black_is_shadow, or alpha_channel"
        )
    public["minimax_network_mode"] = str(
        public["minimax_network_mode"]
    ).strip().lower()
    if public["minimax_network_mode"] not in allowed_minimax_network_modes:
        raise SettingsError(
            "minimax_network_mode must be auto, system_proxy, or direct"
        )
    for key in (
        "minimax_base_url",
        "runninghub_base_url",
        "runninghub_cutout_id",
        "runninghub_input_node_id",
        "runninghub_input_field_name",
        "runninghub_output_node_id",
        "runninghub_hybrid_app_id",
        "runninghub_hybrid_source_node_id",
        "runninghub_hybrid_source_field_name",
        "runninghub_hybrid_mask_node_id",
        "runninghub_hybrid_mask_field_name",
        "runninghub_hybrid_protect_node_id",
        "runninghub_hybrid_protect_field_name",
        "runninghub_hybrid_prompt_node_id",
        "runninghub_hybrid_prompt_field_name",
        "runninghub_hybrid_output_node_id",
        "runninghub_learned_app_id",
        "runninghub_learned_source_node_id",
        "runninghub_learned_source_field_name",
        "runninghub_learned_product_mask_node_id",
        "runninghub_learned_product_mask_field_name",
        "runninghub_learned_geometry_node_id",
        "runninghub_learned_geometry_field_name",
        "runninghub_learned_normal_node_id",
        "runninghub_learned_normal_field_name",
        "runninghub_learned_background_node_id",
        "runninghub_learned_background_field_name",
        "runninghub_learned_prompt_node_id",
        "runninghub_learned_prompt_field_name",
        "runninghub_learned_shadow_output_node_id",
        "runninghub_learned_gain_output_node_id",
        "runninghub_learned_specular_output_node_id",
        "minimax_model",
    ):
        public[key] = str(public[key]).strip()
    for key in ("minimax_base_url", "runninghub_base_url"):
        if not public[key].startswith("https://"):
            raise SettingsError(f"{key} must use HTTPS")
        public[key] = public[key].rstrip("/")
    public["runninghub_enabled"] = bool(public["runninghub_enabled"])
    public["runninghub_hybrid_enabled"] = bool(public["runninghub_hybrid_enabled"])
    public["runninghub_learned_enabled"] = bool(
        public["runninghub_learned_enabled"]
    )

    with _LOCK:
        current = _read()
        secrets = dict(current.get("secrets", {}))
        for name in SECRET_FIELDS:
            clear_value = str(payload.get(name, "") or "").strip()
            if clear_value:
                secrets[name] = protect_secret(clear_value)
            if bool(payload.get(f"remove_{name}", False)):
                secrets.pop(name, None)
        _write({"version": 1, "public": public, "secrets": secrets})
    return get_public_settings()
