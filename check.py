import ee
ee.Initialize(project="cerrado-deforestation")

for t in ee.batch.Task.list()[:5]:
    s = t.status()
    print(s["state"], "|", s.get("description"), "|", s.get("id"))
    if s.get("error_message"):
        print("   error:", s["error_message"])