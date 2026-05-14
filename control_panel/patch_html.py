import re
path = "/home/joseph-ding/.openclaw/workspace/.memory-index/control_panel/templates/base.html"
with open(path, "r") as f:
    content = f.read()

replacement = """      <div class="card span-6">
        <h2>{{ project_name }}</h2>
        <p class="muted">Project path: <code>{{ project_path }}</code></p>
        <p>
          <span class="pill gray">current.md: {{ "yes" if has_current else "no" }}</span>
          <span class="pill gray">milestones: {{ milestone_files|length }}</span>
          <span class="pill gray">daily files: {{ daily_files|length }}</span>
        </p>
      </div>

      <div class="card span-12">
        <h2>Active Memory Folders</h2>
        {% if active_folders %}
          <div class="grid">
          {% for folder, files in active_folders.items() %}
            <div class="card span-4">
              <h4>{{ folder }}</h4>
              <details>
                <summary>{{ files|length }} files</summary>
                <ul>
                {% for file in files %}
                  <li><code>{{ file }}</code></li>
                {% endfor %}
                </ul>
              </details>
            </div>
          {% endfor %}
          </div>
        {% else %}
          <p class="muted">No active folders found.</p>
        {% endif %}
      </div>

      <div class="card span-12">
        <h2>Subprojects</h2>
        {% if subprojects %}
          <div class="grid">
          {% for sp_name, files in subprojects.items() %}
            <div class="card span-4">
              <h4>{{ sp_name }}</h4>
              <details>
                <summary>{{ files|length }} files</summary>
                <ul>
                {% for file in files %}
                  <li><code>{{ file }}</code></li>
                {% endfor %}
                </ul>
              </details>
            </div>
          {% endfor %}
          </div>
        {% else %}
          <p class="muted">No subprojects found.</p>
        {% endif %}
      </div>"""

content = re.sub(r'      <div class="card span-6">\s*<h2>\{\{ project_name \}\}</h2>\s*<p class="muted">Project path: <code>\{\{ project_path \}\}</code></p>\s*<p>\s*<span class="pill gray">current\.md: \{\{ "yes" if has_current else "no" \}\}</span>\s*<span class="pill gray">milestones: \{\{ milestone_files\|length \}\}</span>\s*<span class="pill gray">daily files: \{\{ daily_files\|length \}\}</span>\s*</p>\s*</div>', replacement, content)

with open(path, "w") as f:
    f.write(content)
