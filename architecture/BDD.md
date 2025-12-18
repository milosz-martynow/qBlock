# Block Definition Diagram (BDD)

This file contains a repository-level Block/Class diagram in PlantUML format that describes the key modules, classes and relationships in the `qBlock` project. Render with PlantUML (locally with the `plantuml` CLI or via an online PlantUML server).

## Description

- Shows core classes: `SpinOrbital`, `Orbital`, `SubShell`, `Shell`, `Atom` (defined in `q_block/atom.py`).
- Shows data/helper modules used by `Atom`: `q_block/atoms_data.py`, `q_block/aufbau_exceptions.py`, and `q_block/basis_set_pople.py`.
- Illustrates composition chain: `Atom` → `Shell` → `SubShell` → `Orbital` → `SpinOrbital`.
- Notes how tests and data files relate to the model.

## PlantUML source

![BDD](https://github.com/milosz-martynow/qBlock/blob/main/architecture/BDD.puml)

## How to render

- Locally with PlantUML (requires Java):

```bash
plantuml architecture/BDD.md
```

- Or copy the `@startuml`..`@enduml` block into an online PlantUML renderer (e.g. https://www.plantuml.com/plantuml).

## Relevant paths

- `q_block/atom.py` — core model (SpinOrbital, Orbital, SubShell, Shell, Atom)
- `q_block/atoms_data.py` — atomic symbol/data mappings
- `q_block/aufbau_exceptions.py` — empirical exception mappings
- `q_block/basis_set_pople.py` — basis-set parsing and Regions structure
- `data/basis_set/` — `.gbs` basis files used by population routines
- `tests/test_atom.py` — test coverage and golden-reference checks

If you want a rendered PNG or a Mermaid alternative, I can produce that next.