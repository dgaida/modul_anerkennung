from pydantic import BaseModel, Field
from typing import List, Optional


class ModuleAnalysis(BaseModel):
    """Result of an LLM analysis of an external module description."""

    name: str = Field(..., description="Name of the external module")
    ects: Optional[float] = Field(None, description="Number of ECTS credits")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")


class ModuleMetadata(BaseModel):
    """Metadata for an internal module (from Mocogi)."""

    title: str
    ects: Optional[float] = None
    id: Optional[str] = None


class ComparisonReport(BaseModel):
    """Result of an LLM comparison between an external and internal module."""

    decision: str = Field(..., description="Decision: Ja, Nein, or Vielleicht")
    reasoning: str = Field(..., description="Brief reasoning for the decision")
    report: str = Field(..., description="Detailed comparison report")
