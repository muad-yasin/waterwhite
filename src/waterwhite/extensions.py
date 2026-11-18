"""The six extensions and their switches (plan 1.4, Section 7).

Each extension is off unless its switch in the store's ``setting`` table says "on".
While an extension is off, the tool behaves as the last column of the table in plan
1.4 says: a request that needs it is refused with the named error "extension off",
and no code path of the extension runs. The one exception is a published model the store
holds: a query or a listing that names it answers "not covered", reason "not registered",
with a note that the extension is off (plan 1.4, 3.1, Section 7).
"""

from __future__ import annotations

from .errors import ExtensionOff, Refused

CONTENT_BAN = "content-ban"
GENERATING_MODEL = "generating-model"
FLAGGED_RECORDS = "flagged-records"
PUBLISHED_MODELS = "published-models"
EXPOSURE_COUNTS = "exposure-counts"
ADAPTERS = "adapters"

EXTENSIONS = (
    CONTENT_BAN,
    GENERATING_MODEL,
    FLAGGED_RECORDS,
    PUBLISHED_MODELS,
    EXPOSURE_COUNTS,
    ADAPTERS,
)

SETTING_PREFIX = "extension."

ON = "on"
OFF = "off"
SWITCH_VALUES = (ON, OFF)


def setting_key(extension: str) -> str:
    if extension not in EXTENSIONS:
        raise ValueError(f"unknown extension: {extension!r}")
    return SETTING_PREFIX + extension


def _refuse_broken(extension: str, value: str) -> None:
    raise Refused(
        f"the switch of the extension {extension!r} holds {value!r}; a switch is "
        f"{ON!r} or {OFF!r}",
        extension=extension,
    )


def check_switches(settings: dict[str, str]) -> None:
    """Refuse when the switch of any of the six extensions holds a value that is
    neither "on" nor "off", and refuse a setting under the switch prefix that names no
    extension of this build (a switch the tool cannot read would be taken for off). A
    missing row is off and is not broken. Every answer
    makes this check before anything else (the request, the proof, the listing, the
    report), so a switch file that cannot be read stops all of them, not only the
    answers that happen to read the broken switch. Writing the setting
    (``Store.set_setting``) does not pass through it: that is how the file is fixed."""
    for key in settings:
        if key.startswith(SETTING_PREFIX) and key[len(SETTING_PREFIX):] not in EXTENSIONS:
            raise Refused(
                f"the setting {key!r} names no extension of this build; a switch belongs to one "
                f"of the six extensions and is {ON!r} or {OFF!r}",
                setting=key,
            )
    for extension in EXTENSIONS:
        value = settings.get(setting_key(extension), OFF)
        if value not in SWITCH_VALUES:
            _refuse_broken(extension, value)


def is_on(settings: dict[str, str], extension: str) -> bool:
    """True when the switch is on. A missing row means off. A switch that holds
    anything but "on" or "off" is not read as either: the store writes no such
    value, and an answer must not depend on a guess at what was meant, so the
    request is refused."""
    value = settings.get(setting_key(extension), OFF)
    if value not in SWITCH_VALUES:
        _refuse_broken(extension, value)
    return value == ON


def require(settings: dict[str, str], extension: str, what: str) -> None:
    """Refuse a request that needs an extension that is off (plan 2.3, L7)."""
    if not is_on(settings, extension):
        raise ExtensionOff(f"{what} needs the extension {extension!r}, which is off",
                           extension=extension)
