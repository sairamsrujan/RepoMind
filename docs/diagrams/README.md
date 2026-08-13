# Diagrams

Every diagram for the report, the slide deck and the README. All hand-laid SVG —
no rendering step, no toolchain, nothing to install. Open any file in a browser,
or insert it directly into Word or PowerPoint (both support SVG).

## System architecture

The same architecture at three aspect ratios. Pick by where it is going.

| File | Size | Use for |
|---|---|---|
| [`architecture.svg`](architecture.svg) | 854×676 | the README — fits GitHub's content width with no downscaling |
| [`architecture-slide.svg`](architecture-slide.svg) | 1280×720 | a 16:9 presentation slide |
| [`architecture-a4.svg`](architecture-a4.svg) | 900×1280 | A4 portrait, for the written report |

## UML

| File | Diagram | Shows |
|---|---|---|
| [`use-case.svg`](use-case.svg) | Use case | Actors and what they can do. Refusal is a use case, not an error path. |
| [`class-diagram.svg`](class-diagram.svg) | Class | The real classes, attributes and methods, extracted from the source. |
| [`sequence-query.svg`](sequence-query.svg) | Sequence | One question, end to end, including the guard and the widened retry. |
| [`activity.svg`](activity.svg) | Activity | Index-or-reuse, then ask, then verify. Fork/join for chunking and linking. |
| [`state-indexing.svg`](state-indexing.svg) | State | The `status` field of `manifest.json`, which the UI polls. |
| [`component.svg`](component.svg) | Component | Module boundaries, and the rule that nothing live may import `eval/`. |

## Keeping them honest

The class diagram lists classes and methods that **exist in the code**. If you
rename `Retriever.rrf_fuse` or drop a field from `AnswerResult`, the diagram is
wrong and an examiner reading both will find it. Re-extract with:

```bash
.venv/bin/python -c "import ast,pathlib; [print(p, [n.name for n in ast.parse(p.read_text()).body if isinstance(n, ast.ClassDef)]) for p in pathlib.Path('.').glob('*/[a-z]*.py')]"
```

## Two traps when editing

**Band backgrounds are opaque.** Anything drawn *before* a band rect that
overlaps it gets painted over. Connectors that cross into a band must stay at
the end of the file. All three architecture files once had this: an arrow was
half-invisible and its label gone entirely, which read as a missing connection
rather than a z-order mistake.

**`--` is illegal inside an XML comment.** It does not warn — the file simply
stops being valid XML and every renderer shows a broken image. Validate after
editing:

```bash
python3 -c "import xml.dom.minidom,glob; [xml.dom.minidom.parse(f) for f in glob.glob('docs/diagrams/*.svg')]; print('all valid')"
```
