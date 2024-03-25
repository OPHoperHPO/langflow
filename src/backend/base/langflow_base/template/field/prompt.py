from typing import Optional

from langflow_base.template.field.base import TemplateField


class DefaultPromptField(TemplateField):
    name: str
    display_name: Optional[str] = None
    field_type: str = "str"

    advanced: bool = False
    multiline: bool = True
    input_types: list[str] = ["Document", "BaseOutputParser", "Text", "Record"]
    value: str = ""  # Set the value to empty string
