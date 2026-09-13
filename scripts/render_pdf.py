import subprocess
import os

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_path):
    chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

html_url = "file:///D:/Projects/RAP/docs/live_demo_results.html"

out_pdf_1 = r"D:\Projects\RAP\Deliverables\Live Demonstration & Inference Results.pdf"
out_pdf_2 = r"D:\Projects\RAP\Deliverables\05_Live_Demonstration_and_Inference_Results.pdf"

cmd1 = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--no-pdf-header-footer",
    f"--print-to-pdf={out_pdf_1}",
    html_url
]

cmd2 = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--no-pdf-header-footer",
    f"--print-to-pdf={out_pdf_2}",
    html_url
]

print("Rendering PDF 1...")
res1 = subprocess.run(cmd1, capture_output=True, text=True)
print("Code 1:", res1.returncode)
print("Output 1:", res1.stdout, res1.stderr)

print("Rendering PDF 2...")
res2 = subprocess.run(cmd2, capture_output=True, text=True)
print("Code 2:", res2.returncode)

if os.path.exists(out_pdf_1):
    print(f"Created {out_pdf_1} ({os.path.getsize(out_pdf_1)} bytes)")
if os.path.exists(out_pdf_2):
    print(f"Created {out_pdf_2} ({os.path.getsize(out_pdf_2)} bytes)")
