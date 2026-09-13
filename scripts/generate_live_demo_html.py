import base64
import os

def get_b64(path):
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode('utf-8')

img1_b64 = get_b64('D:/Projects/RAP/docs/demo_images/demo_case_1_compliant_live.png')
img2_b64 = get_b64('D:/Projects/RAP/docs/demo_images/demo_case_2_violation_live.png')
img3_b64 = get_b64('D:/Projects/RAP/docs/demo_images/demo_case_3_audit_log_live.png')
img_scalability_b64 = get_b64('D:/Projects/RAP/Deliverables/Scalability.png')

html_content = f"""<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Live Demonstration &amp; Empirical Results Showcase</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

  <style>
    /* ============================================================
       EXECUTIVE 2-PAGE A4 LIVE DEMONSTRATION & TELEMETRY REPORT
       ============================================================ */
    @page {{
      size: A4 portrait;
      margin: 8mm 10mm 8mm 10mm;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #0f172a;
      background: #090d16;
      line-height: 1.35;
      font-size: 8pt;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
      min-height: 100vh;
    }}

    /* Screen Top Toolbar */
    .screen-toolbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding: 10px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: #f8fafc;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }}

    .toolbar-left {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .toolbar-icon {{
      width: 28px;
      height: 28px;
      background: linear-gradient(135deg, #0284c7, #38bdf8);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
    }}

    .toolbar-text h1 {{
      font-size: 13px;
      font-weight: 700;
      color: #f8fafc;
    }}

    .toolbar-right {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .status-pill {{
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34d399;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 10px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 5px;
    }}

    .status-pill::before {{
      content: '';
      display: inline-block;
      width: 6px;
      height: 6px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 6px #10b981;
    }}

    .print-btn {{
      background: #0284c7;
      color: #ffffff;
      border: none;
      padding: 5px 14px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 11px;
      cursor: pointer;
    }}

    /* Main Container */
    .document-container {{
      max-width: 210mm;
      margin: 20px auto 40px auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    /* Individual A4 Sheet (Print Page) */
    .a4-sheet {{
      width: 210mm;
      min-height: 297mm;
      height: 297mm;
      max-height: 297mm;
      background: #ffffff;
      margin: 0 auto;
      padding: 7mm 9mm 7mm 9mm;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      box-sizing: border-box;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      border-radius: 2px;
      page-break-after: always;
      page-break-inside: avoid;
      overflow: hidden;
    }}

    .sheet-body {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    /* Header Styling */
    .memo-header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f8fafc;
      padding: 8px 12px;
      border-radius: 6px;
      border-left: 4px solid #0284c7;
    }}

    .memo-header-top {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 3px;
    }}

    .memo-title {{
      font-size: 12pt;
      font-weight: 800;
      letter-spacing: -0.3px;
      color: #ffffff;
    }}

    .meta-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }}

    .badge-item {{
      background: rgba(255, 255, 255, 0.05);
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}

    .badge-label {{
      color: #94a3b8;
      font-size: 6.2pt;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      display: block;
    }}

    .badge-val {{
      color: #f1f5f9;
      font-weight: 600;
      font-size: 7pt;
    }}

    .links-strip {{
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-left: 3px solid #16a34a;
      border-radius: 5px;
      padding: 4px 8px;
      font-size: 7.2pt;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .links-strip a {{
      color: #0284c7;
      text-decoration: underline;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
    }}

    /* Section Headings */
    h2 {{
      font-size: 8.6pt;
      font-weight: 700;
      color: #0f172a;
      border-bottom: 1.5px solid #0284c7;
      padding-bottom: 2px;
      margin-top: 3px;
      margin-bottom: 1px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .tag-pill {{
      font-size: 6.5pt;
      font-family: 'JetBrains Mono', monospace;
      padding: 1px 6px;
      border-radius: 3px;
      font-weight: 600;
    }}

    .pill-green {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
    .pill-red {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .pill-blue {{ background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }}
    .pill-amber {{ background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }}

    /* Case Card Layout */
    .case-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 6px 8px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .case-container-2col {{
      display: grid;
      grid-template-columns: 290px 1fr;
      gap: 10px;
      align-items: center;
    }}

    .case-img-frame {{
      width: 100%;
      height: 142px;
      border-radius: 5px;
      overflow: hidden;
      border: 1px solid #cbd5e1;
      background: #090d16;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
    }}

    .case-img-frame img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #090d16;
      display: block;
    }}

    .case-details {{
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}

    .metric-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }}

    .badge-chip {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 6.6pt;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      padding: 1px 5px;
      border-radius: 3px;
      font-weight: 600;
      color: #0f172a;
    }}

    .chip-green {{ background: #ecfdf5; border-color: #10b981; color: #047857; }}
    .chip-red {{ background: #fef2f2; border-color: #ef4444; color: #b91c1c; }}
    .chip-blue {{ background: #f0f9ff; border-color: #0284c7; color: #0369a1; }}

    .reasoning-box {{
      background: #ffffff;
      border-left: 3px solid #10b981;
      padding: 4px 7px;
      border-radius: 0 4px 4px 0;
      border-top: 1px solid #e2e8f0;
      border-right: 1px solid #e2e8f0;
      border-bottom: 1px solid #e2e8f0;
    }}

    .reasoning-box.violation {{
      border-left-color: #ef4444;
    }}

    .reasoning-box.warning {{
      border-left-color: #f59e0b;
    }}

    .q-text {{
      font-size: 6.8pt;
      color: #64748b;
      font-weight: 600;
      margin-bottom: 1px;
    }}

    .ans-text {{
      font-size: 7.2pt;
      font-weight: 600;
      line-height: 1.3;
    }}

    .ans-text.compliant {{ color: #15803d; }}
    .ans-text.violation {{ color: #b91c1c; }}
    .ans-text.warning {{ color: #b45309; }}

    .telemetry-meta {{
      display: flex;
      justify-content: space-between;
      font-family: 'JetBrains Mono', monospace;
      font-size: 6.3pt;
      color: #64748b;
      margin-top: 2px;
      padding-top: 2px;
      border-top: 1px dashed #e2e8f0;
    }}

    /* ============================================================
       PAGE 2: HISTORY CARD & AUDIT BREAKDOWN STYLES
       ============================================================ */
    .history-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 5px 7px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}

    .history-img-frame {{
      width: 100%;
      height: 90px;
      border-radius: 5px;
      overflow: hidden;
      border: 1px solid #1e293b;
      background: #090d16;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }}

    .history-img-frame img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #090d16;
      display: block;
    }}

    .history-events-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
    }}

    .history-event-card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 3px 5px;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }}

    .history-event-card.event-violation {{
      border-left: 3px solid #ef4444;
    }}

    .history-event-card.event-scan {{
      border-left: 3px solid #0284c7;
    }}

    .history-event-card.event-guardrail {{
      border-left: 3px solid #f59e0b;
    }}

    .event-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 6pt;
    }}

    .event-time {{
      color: #64748b;
      font-weight: 700;
    }}

    .event-badge {{
      padding: 1px 4px;
      border-radius: 2px;
      font-size: 5.6pt;
      font-weight: 700;
      text-transform: uppercase;
    }}

    .event-badge.red {{ background: #fee2e2; color: #b91c1c; }}
    .event-badge.blue {{ background: #e0f2fe; color: #0369a1; }}
    .event-badge.amber {{ background: #fef3c7; color: #b45309; }}

    .event-q {{
      font-size: 6.5pt;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
    }}

    .event-ans {{
      font-size: 6.1pt;
      line-height: 1.2;
      color: #334155;
    }}

    .history-meta-note {{
      font-size: 6.5pt;
      color: #475569;
      line-height: 1.2;
      background: #f1f5f9;
      padding: 2px 5px;
      border-radius: 3px;
      border: 1px solid #e2e8f0;
    }}

    /* Table Styling */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 6.8pt;
      margin: 1px 0;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      overflow: hidden;
    }}

    th, td {{
      padding: 2px 5px;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }}

    th {{
      background: #f1f5f9;
      font-weight: 700;
      color: #334155;
      text-transform: uppercase;
      font-size: 6.3pt;
      letter-spacing: 0.3px;
    }}

    tr:nth-child(even) {{
      background: #f8fafc;
    }}

    /* Scalability Infographic Container */
    .scalability-container {{
      width: 100%;
      border-radius: 5px;
      overflow: hidden;
      border: 1px solid #cbd5e1;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
      background: #ffffff;
      display: flex;
      justify-content: center;
      align-items: center;
    }}

    .scalability-img {{
      width: 100%;
      height: auto;
      max-height: 420px;
      object-fit: contain;
      display: block;
    }}

    /* Footer */
    .sheet-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid #e2e8f0;
      padding-top: 3px;
      font-size: 6.7pt;
      color: #94a3b8;
    }}

    .footer-left {{
      font-weight: 600;
      color: #64748b;
    }}

    /* ============================================================
       PRINT MEDIA OPTIMIZATION: STRICT 2 A4 PAGES
       ============================================================ */
    @media print {{
      @page {{
        size: A4 portrait;
        margin: 8mm 10mm 8mm 10mm;
      }}

      html, body {{
        background: #ffffff !important;
        padding: 0 !important;
        margin: 0 !important;
      }}

      .screen-toolbar {{
        display: none !important;
      }}

      .document-container {{
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
        display: block !important;
      }}

      .a4-sheet {{
        width: 100% !important;
        max-width: 100% !important;
        height: 279mm !important;
        max-height: 279mm !important;
        min-height: unset !important;
        margin: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        page-break-after: always !important;
        break-after: page !important;
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
      }}

      .sheet-body {{
        gap: 4px !important;
        flex: 1 !important;
        justify-content: flex-start !important;
      }}

      #page-2 {{
        page-break-after: avoid !important;
        break-after: avoid !important;
      }}
    }}
  </style>
</head>

<body>

  <!-- Top Web App Navigation Toolbar -->
  <header class="screen-toolbar">
    <div class="toolbar-left">
      <div class="toolbar-icon">&#128737;</div>
      <div class="toolbar-text">
        <h1>Live Demonstration &amp; Telemetry Report (2-Page Executive Brief)</h1>
      </div>
    </div>
    <div class="toolbar-right">
      <span class="status-pill">Live Hugging Face Space Connected</span>
      <button class="print-btn" onclick="window.print()">Export / Print PDF</button>
    </div>
  </header>

  <!-- Main Document Container -->
  <main class="document-container">

    <!-- ========================================================
         PAGE 1: TEST CASE 1 & TEST CASE 2
         ======================================================== -->
    <article class="a4-sheet" id="page-1">
      <div class="sheet-body">

        <!-- Executive Header -->
        <header class="memo-header">
          <div class="memo-header-top">
            <span class="memo-title">Industrial PPE Safety Detection &amp; Reasoning API</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 7pt; color: #38bdf8;">PORTFOLIO DEMONSTRATION &bull; DELIVERABLE 05</span>
          </div>
          <div class="meta-badges">
            <div class="badge-item">
              <span class="badge-label">Architecture</span>
              <span class="badge-val">RT-DETR-L + Groq LLaMA-3.3-70B</span>
            </div>
            <div class="badge-item">
              <span class="badge-label">Active Space</span>
              <span class="badge-val">akash4303/worker-ppe-detection-reasoning</span>
            </div>
            <div class="badge-item">
              <span class="badge-label">Confidence Cutoff</span>
              <span class="badge-val">0.45 (Unified Across System)</span>
            </div>
            <div class="badge-item">
              <span class="badge-label">Reasoning Engine</span>
              <span class="badge-val">Zero Frameworks (Pure Groq SDK)</span>
            </div>
          </div>
        </header>

        <!-- Live Deployment Strip -->
        <div class="links-strip">
          <span><strong>Live Web UI:</strong> <a href="https://akash4303-worker-ppe-detection-reasoning.hf.space" target="_blank">https://akash4303-worker-ppe-detection-reasoning.hf.space</a></span>
          <span><strong>Vercel Mirror:</strong> <a href="https://industrial-ppe-detection.vercel.app" target="_blank">industrial-ppe-detection.vercel.app</a></span>
        </div>

        <!-- Case 1: Compliant Case -->
        <h2>
          <span>1. Test Case 1: Full PPE Compliance (Person + Helmet + Vest)</span>
          <span class="tag-pill pill-green">Compliant &bull; 200 OK</span>
        </h2>
        <div class="case-card">
          <div class="case-container-2col">
            <div class="case-img-frame">
              <img src="{img1_b64}" alt="Case 1: Fully Compliant Worker Live UI Screenshot">
            </div>
            <div class="case-details">
              <div class="metric-badges">
                <span class="badge-chip chip-green">Workers: 1 (95.0%)</span>
                <span class="badge-chip chip-green">Helmets: 1 (86.0%)</span>
                <span class="badge-chip chip-green">Vests: 1 (95.6%)</span>
                <span class="badge-chip chip-blue">Sharpness: 541.6</span>
              </div>
              <div class="reasoning-box">
                <div class="q-text">Query: "Is he wearing helmet or not ?"</div>
                <div class="ans-text compliant">
                  &#10003; VERIFIED COMPLIANCE ASSESSMENT: Yes, the worker is wearing a helmet. Detection output confirms 1 worker present (confidence: 0.95), with 1 helmet/hard-hat detected (hat: 1, confidence: 0.86).
                </div>
                <div class="telemetry-meta">
                  <span>Model: llama-3.3-70b-versatile</span>
                  <span>Reasoning Latency: 577.44 ms</span>
                  <span>Device History: #13</span>
                </div>
              </div>
              <p style="font-size: 6.9pt; color: #475569; line-height: 1.25;">
                <strong>Technical Analysis:</strong> The vision transformer successfully identified the worker in a complex industrial workshop setting. Both hard-hat (86.0% conf) and safety vest (95.6% conf) exceed the 0.45 cutoff. The hand-written reasoning engine matched bounding-box geometries and synthesized an authoritative compliance confirmation via Groq in under 580 ms.
              </p>
            </div>
          </div>
        </div>

        <!-- Case 2: Violation Case -->
        <h2>
          <span>2. Test Case 2: Active Safety Violation (Missing Helmet &amp; Safety Vest)</span>
          <span class="tag-pill pill-red">Violation &bull; 200 OK</span>
        </h2>
        <div class="case-card">
          <div class="case-container-2col">
            <div class="case-img-frame">
              <img src="{img2_b64}" alt="Case 2: Safety Violation Live UI Screenshot">
            </div>
            <div class="case-details">
              <div class="metric-badges">
                <span class="badge-chip chip-green">Workers: 1 (88.8%)</span>
                <span class="badge-chip chip-red">Helmets: 0 (0.0%)</span>
                <span class="badge-chip chip-red">Vests: 0 (0.0%)</span>
                <span class="badge-chip chip-blue">Sharpness: 706.0</span>
              </div>
              <div class="reasoning-box violation">
                <div class="q-text">Query: "Is he wearing helmet or not ?"</div>
                <div class="ans-text violation">
                  &#9888; VERIFIED COMPLIANCE ASSESSMENT: No, he is not wearing a helmet. Detection output confirms 1 worker present (confidence: 0.89), but zero helmets/hard-hats detected (hat: 0). This constitutes an OSHA PPE safety violation.
                </div>
                <div class="telemetry-meta">
                  <span>Model: llama-3.3-70b-versatile</span>
                  <span>Reasoning Latency: 564.33 ms</span>
                  <span>Device History: #17</span>
                </div>
              </div>
              <p style="font-size: 6.9pt; color: #475569; line-height: 1.25;">
                <strong>Technical Analysis:</strong> The worker is photographed from behind with head turned away. RT-DETR detects the person with 88.8% confidence but zero helmets and zero vests. The guardrail layer verifies non-zero worker presence and passes the context to Groq, which triggers an immediate OSHA safety non-compliance violation notice.
              </p>
            </div>
          </div>
        </div>

      </div>

      <!-- Page 1 Footer -->
      <footer class="sheet-footer">
        <div class="footer-left">Industrial PPE Safety Vision &amp; Reasoning &bull; Live Empirical Demonstration</div>
        <div class="footer-right">Page 1 of 2</div>
      </footer>
    </article>

    <!-- ========================================================
         PAGE 2: LIVE AUDIT LOG, TELEMETRY & SCALABILITY ROADMAP
         ======================================================== -->
    <article class="a4-sheet" id="page-2">
      <div class="sheet-body">

        <!-- Case 3: Device Activity History & Audit Trail -->
        <h2>
          <span>3. Device Activity History &amp; Immutable Audit Trail</span>
          <span class="tag-pill pill-amber">Live Client Log &bull; 3 Logged Events</span>
        </h2>
        <div class="history-card">
          <!-- Full-Width Live History Screenshot -->
          <div class="history-img-frame">
            <img src="{img3_b64}" alt="Device Activity History Live UI Audit Log Screenshot">
          </div>

          <!-- Matching Breakdown of the 3 Recorded Events -->
          <div class="history-events-grid">
            <!-- Event 1 -->
            <div class="history-event-card event-violation">
              <div class="event-top">
                <span class="event-time">12:27</span>
                <span class="event-badge red">OSHA Violation</span>
              </div>
              <div class="event-q">Q: "Is he wearing helmet or not ?"</div>
              <div class="event-ans">
                <strong>Result:</strong> No, he is not wearing a helmet. 1 worker present (conf: 0.96), 0 helmets detected. Constitutes OSHA PPE violation.
              </div>
            </div>

            <!-- Event 2 -->
            <div class="history-event-card event-scan">
              <div class="event-top">
                <span class="event-time">12:26</span>
                <span class="event-badge blue">Detection Scan</span>
              </div>
              <div class="event-q">Raw Detector Output (1 object)</div>
              <div class="event-ans">
                <strong>Counts:</strong> Workers: 1 | Hats: 0 | Vests: 0. RT-DETR-L baseline inference confirmed worker present without safety gear.
              </div>
            </div>

            <!-- Event 3 -->
            <div class="history-event-card event-guardrail">
              <div class="event-top">
                <span class="event-time">12:26</span>
                <span class="event-badge amber">Guardrail Trigger</span>
              </div>
              <div class="event-q">Q: "Is he wearing helmet or not ?"</div>
              <div class="event-ans">
                <strong>Refusal:</strong> Insufficient information: No workers or safety equipment detected with confidence above the 0.45 threshold.
              </div>
            </div>
          </div>

          <!-- Audit Mechanism Note -->
          <div class="history-meta-note">
            <strong>Audit Mechanism &amp; Integrity:</strong> The client-side <em>Device Activity History</em> maintains an immutable chronological record of every scan, reasoning assessment, and deterministic guardrail refusal. When low-confidence artifacts fall below 0.45, the pre-LLM Guardrail Layer immediately halts synthesis to prevent hallucination.
          </div>
        </div>

        <!-- Section 4: Live Telemetry Matrix -->
        <h2>
          <span>4. Live Production Telemetry &amp; Verification Matrix</span>
          <span class="tag-pill pill-blue">Empirical Benchmark Data</span>
        </h2>
        <table>
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Detected Entities</th>
              <th>Worker Conf.</th>
              <th>Laplacian Blur</th>
              <th>Guardrail Decision</th>
              <th>LLM Latency</th>
              <th>Audit Verdict</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Case 1: Full PPE</strong></td>
              <td>1 Worker, 1 Hat, 1 Vest</td>
              <td>95.0% (Valid)</td>
              <td>541.6 (Sharp)</td>
              <td>Passed (&ge; 0.45)</td>
              <td>577.44 ms</td>
              <td><span class="tag-pill pill-green">&#10003; Compliant</span></td>
            </tr>
            <tr>
              <td><strong>Case 2: Missing Gear</strong></td>
              <td>1 Worker, 0 Hats, 0 Vests</td>
              <td>88.8% (Valid)</td>
              <td>706.0 (Sharp)</td>
              <td>Passed (&ge; 0.45)</td>
              <td>564.33 ms</td>
              <td><span class="tag-pill pill-red">&#9888; OSHA Violation</span></td>
            </tr>
            <tr>
              <td><strong>Case 3: Below Cutoff</strong></td>
              <td>0 Objects Above 0.45</td>
              <td>&lt; 0.45 (Filtered)</td>
              <td>N/A (Sub-threshold)</td>
              <td>Triggered (Refusal)</td>
              <td>0.00 ms (Bypassed)</td>
              <td><span class="tag-pill pill-amber">&#8856; Insufficient Info</span></td>
            </tr>
          </tbody>
        </table>

        <!-- Section 5: Scalability & Limitations (Image 2 Infographic) -->
        <div class="scalability-container">
          <img src="{img_scalability_b64}" alt="5. Scalability &amp; Limitations - Current Setup vs. Production Scale Infographic" class="scalability-img">
        </div>

      </div>

      <!-- Page 2 Footer -->
      <footer class="sheet-footer">
        <div class="footer-left">Industrial PPE Safety Vision &amp; Reasoning &bull; Live Empirical Demonstration</div>
        <div class="footer-right">Page 2 of 2</div>
      </footer>
    </article>

  </main>

</body>

</html>
"""

with open('D:/Projects/RAP/docs/live_demo_results.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Generated docs/live_demo_results.html successfully! Size: {len(html_content)} bytes")
