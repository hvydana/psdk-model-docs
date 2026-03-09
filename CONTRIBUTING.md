# Contributing to Model Documentation

## Structure

This repository follows a structured organization:

```
model_documentation/
├── models/
│   ├── robotics/          # Robotics-specific models
│   ├── industrial/        # Industrial-specific models
│   ├── healthcare/        # Healthcare-specific models
│   └── shared/            # Models used across segments
├── systems/               # End-to-end systems and pipelines
├── inference-frameworks/  # Inference runtimes and frameworks
├── README.md              # Repository overview
└── MODELS_INDEX.md        # Complete index of all content
```

## Adding New Models

1. Create a folder in the appropriate segment directory
2. Add a `README.md` with the following structure:
   - Overview
   - Segment(s)
   - Use Cases
   - Description
   - Runtime Support
   - Source reference

## Adding New Systems

1. Create a folder in `systems/`
2. Document components, pipeline flow, and integration points
3. Link to relevant models and frameworks

## Updating Existing Documentation

- Keep descriptions concise and technical
- Include runtime and hardware support information
- Reference source materials
- Update MODELS_INDEX.md when adding new entries

## Format Guidelines

- Use clear headings
- Include code blocks for pipeline flows
- Keep consistent formatting across all README files
- Link between related models and systems
