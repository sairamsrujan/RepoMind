# Diagrams

Every diagram for the report, the slide deck and the README. All hand-laid SVG —
no rendering step, no toolchain, nothing to install. Open any file in a browser,
or insert it directly into Word or PowerPoint (both support SVG).

## System architecture

The same architecture at three aspect ratios. Pick by where it is going.

| File | Size | Use for |
|---|---|---|
| [`architecture.svg`](architecture.svg) | 854×676 | the README — fits GitHub's content width with no downscaling |
| [`architecture-slide.svg`](architecture-slide.svg) | 1280×720 | a 16:9 presentation slide (system architecture) |
| [`architecture-a4.svg`](architecture-a4.svg) | 900×1280 | A4 portrait, for the written report |
| [`pipeline-slide.svg`](pipeline-slide.svg) | 1280×720 | a 16:9 presentation slide (complete pipeline flowchart) |
| [`pipeline-detailed.svg`](pipeline-detailed.svg) | 1180×2210 | the whole pipeline in one top-to-bottom flowchart, with the module, model and constant behind every step |

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

## Four traps when editing

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

**An arrowhead that stops short reads as pointing at nothing.** Terminate every
arrow *on* the target's edge, and away from its corners. Three were wrong: one
stopped 22px above its box, one landed on a box's top-left corner so it looked
like it pointed into the gap between two stacked boxes, and one ended with a 4px
sideways jog that turned the arrowhead 90° into empty space. None of these look
like mistakes at thumbnail size, which is why they survived.

**Text has no background.** A label drawn over a lifeline, an activation bar or
a filled tag is not clipped — it just overlaps, and at a glance the reader takes
it for a rendering artefact. Check labels against the boxes near them, not only
against other labels.

Both classes are found by measuring rather than looking: compute each label's
width with the real font metrics (`PIL.ImageFont`, Helvetica Regular/Bold —
`SFNS.ttf` is variable and `set_variation_by_axes` overshoots bold by ~27%,
which invents overflows that are not there), then test every label box against
every other box, and every arrow endpoint against its target's edge.
