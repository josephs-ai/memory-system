import sys

path = "/home/joseph-ding/.openclaw/workspace/.memory-index/control_panel/views.py"
with open(path, "r") as f:
    content = f.read()

new_detail = """@router.get("/projects/{project_name}")
def project_detail(request: Request, project_name: str):
    p = PROJECTS_DIR / project_name
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    active_dir = p / "active"
    folders = {}
    subprojects = {}
    
    if active_dir.exists():
        for sub in active_dir.iterdir():
            if sub.is_dir():
                if sub.name == "subprojects":
                    for sp in sub.iterdir():
                        if sp.is_dir():
                            subprojects[sp.name] = [str(f.name) for f in sp.rglob("*.*") if f.is_file()]
                else:
                    folders[sub.name] = [str(f.name) for f in sub.rglob("*.*") if f.is_file()]

    current_file = p / "current.md"
    milestone_files = sorted(p.glob("milestone-*.md"))
    daily_dir = p / "daily"
    daily_files = sorted(daily_dir.glob("*.md")) if daily_dir.exists() else []

    return render(
        request,
        f"Project: {project_name}",
        "project_detail",
        project_name=project_name,
        project_path=str(p),
        has_current=current_file.exists(),
        milestone_files=[str(x.name) for x in milestone_files],
        daily_files=[str(x.name) for x in daily_files[-15:]],
        current_file=str(current_file),
        active_folders=folders,
        subprojects=subprojects
    )"""

import re
content = re.sub(r'@router\.get\("/projects/\{project_name\}"\).*?return render\([^)]+\)', new_detail, content, flags=re.DOTALL)

with open(path, "w") as f:
    f.write(content)
