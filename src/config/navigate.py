from pathlib import Path
from typing import Optional


class ProjectRouter:
    def __init__(
        self, project_path: Path = Path(__file__).parents[1], stor_path: Optional[Path] = None
    ):
        self.project_path = project_path
        self.stor_path = project_path / "stor" if stor_path is None else stor_path
        self.data_path = self.stor_path / "data"
        self.models_path = self.stor_path / "models"


project_router = ProjectRouter()
