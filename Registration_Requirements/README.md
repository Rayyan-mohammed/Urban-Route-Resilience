# Registration Requirements — Route Resilience (BAH 2026)

This folder contains everything required for the **Registration & Idea/Concept
Submission** stage of the **Bharatiya Antariksh Hackathon 2026** (ISRO ×
Hack2skill), due **1 July 2026**.

All content is kept in Markdown (`.md`). If the portal requires a PDF or DOCX,
export the relevant file — the text here is the single source of truth.

## Contents

| File | Purpose | Priority |
|---|---|---|
| [00_Registration_Checklist.md](00_Registration_Checklist.md) | Registration steps + team/portal info to fill in | Must |
| [01_Proposal.md](01_Proposal.md) | **The proposal document** — all required sections | Highest |
| [02_Diagrams/System_Architecture.md](02_Diagrams/System_Architecture.md) | End-to-end system architecture diagram | Recommended |
| [02_Diagrams/Pipeline_Flowchart.md](02_Diagrams/Pipeline_Flowchart.md) | Four-phase pipeline flowchart | Recommended |
| [02_Diagrams/Workflow_Diagram.md](02_Diagrams/Workflow_Diagram.md) | Team + timeline workflow diagram | Recommended |
| [02_Diagrams/Dashboard_Mockup.md](02_Diagrams/Dashboard_Mockup.md) | Decision-support dashboard mockup | Optional, high-value |
| [03_References.md](03_References.md) | Datasets + literature + event references | Supporting |
| [04_Final_Review_Checklist.md](04_Final_Review_Checklist.md) | Pre-submission QA checklist | Must |

## How to use this before 1 July

1. Fill in the team/portal blanks in `00_Registration_Checklist.md`.
2. Read through `01_Proposal.md`; replace any `<…>` placeholders (team names).
3. Export `01_Proposal.md` (+ diagrams) to PDF if the portal needs uploads.
4. Run the `04_Final_Review_Checklist.md` before submitting.

> Diagrams use [Mermaid](https://mermaid.js.org/), which renders on GitHub, in
> VS Code (with a Mermaid extension), and in most Markdown-to-PDF tools. ASCII
> fallbacks are included so nothing is lost if Mermaid is unavailable.

The deep technical plan behind this submission lives in the repository root:
[`../roadmap.md`](../roadmap.md) (full feasibility study) and
[`../PROJECT_STATE.md`](../PROJECT_STATE.md) (live build status).
