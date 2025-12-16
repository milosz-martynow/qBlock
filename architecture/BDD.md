# Block Definition Diagram (BDD)

This file contains a repository-level Block/Class diagram in PlantUML format that describes the key modules, classes and relationships in the `qBlock` project. Render with PlantUML (locally with the `plantuml` CLI or via an online PlantUML server).

## Description

- Shows core classes: `SpinOrbital`, `Orbital`, `SubShell`, `Shell`, `Atom` (defined in `q_block/atom.py`).
- Shows data/helper modules used by `Atom`: `q_block/atoms_data.py`, `q_block/aufbau_exceptions.py`, and `q_block/basis_set_pople.py`.
- Illustrates composition chain: `Atom` → `Shell` → `SubShell` → `Orbital` → `SpinOrbital`.
- Notes how tests and data files relate to the model.

## PlantUML source

```plantuml
@startuml
' Modules
package "q_block" {
  package "modules" {
    class "SpinOrbital" {
      - n:int
      - l:int
      - m:int
      - s:float
      - occupied:bool
      - data:Any
    }
    class "Orbital" {
      - spin_up:SpinOrbital
      - spin_down:SpinOrbital
    }
    class "SubShell" {
      - n:int
      - l:int
      - orbitals: List<Orbital>
      + capacity(): int
    }
    class "Shell" {
      - n:int
      - subshells: List<SubShell>
    }
    class "Atom" {
      - Z:int
      - shells: Dict<int,Shell>
      - _all_subshells: List<SubShell>
      - basis_set: Optional[Dict]
      + fill_occupancy(): None
      + populate_spinorbitals_with_gto(): None
    }
  }
  package "data & helpers" {
    class "atoms_data" as AD <<module>> {
      + ATOMS_SYMBOLS_Z_TO_SYMBOL: Dict
    }
    class "aufbau_exceptions" as AE <<module>> {
      + EMPIRICAL_EXCEPTIONS: Dict
    }
    class "basis_set_pople" as BSP <<module>> {
      + parse_gaussian_basis(...)
    }
  }
}

' Relationships (composition / uses)
Atom "1" *-- "*" Shell
Shell "1" *-- "*" SubShell
SubShell "1" *-- "*" Orbital
Orbital "1" *-- "2" SpinOrbital

Atom ..> AE : uses EMPIRICAL_EXCEPTIONS
Atom ..> AD : references ATOMS_SYMBOLS_Z_TO_SYMBOL
Atom ..> BSP : uses parsing & basis assignment

' Top-level scripts & tests
package "root" {
  class "main.py" <<script>>
  class "tests/test_atom.py" <<tests>>
  class "data/basis_set" <<folder>>
}

main.py ..> Atom
tests/test_atom.py ..> Atom
data/basis_set ..> BSP

@enduml
```

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