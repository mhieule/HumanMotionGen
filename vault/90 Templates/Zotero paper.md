---
citekey: "{{citekey}}"
title: "{{title}}"
authors: "{% for c in creators %}{{c.firstName}} {{c.lastName}}{% if not loop.last %}, {% endif %}{% endfor %}"
year: {{date | format("YYYY")}}
venue: "{{publicationTitle or proceedingsTitle or repository}}"
url: "{{url}}"
tags: [paper{% for t in tags %}, {{t.tag}}{% endfor %}]
---
{# ============================================================
   Zotero Integration import template (Nunjucks).
   Used ONLY by the Zotero Integration plugin. Do not insert it
   with Obsidian's "Insert template" command.
   {{ x }}      -> prints a Zotero field
   {% ... %}    -> logic (if / for / persist)
   curly-brace + hash  -> comment (like this block), never appears in the note
   Tip: run "Zotero Integration: Data explorer" to see every
   field available for a paper.
   Frontmatter is rewritten on every import: add course tags
   (e.g. "seminar", no spaces) to the item in Zotero, not here.
   ============================================================ #}
# {{title}}

> [!info] Source
> [Open in Zotero]({{desktopURI}}){% for at in attachments | filterby("path", "endswith", ".pdf") %} · [PDF]({{at.desktopURI | replace("/select/", "/open-pdf/")}}){% endfor %}{% if url %} · [Web]({{url}}){% endif %}

{# Everything inside "persist" survives re-imports. The headings are
   written only on the first import; after that your text is kept. #}
{% persist "notes" %}{% if isFirstImport %}
status:: to-read

## TL;DR
- 

## Problem & motivation

## Method

## What's new vs. prior work

## Results

## Limitations & my critique

## Questions

## Connections
- Concepts: 
- Related papers: 
{% endif %}{% endpersist %}

{# Child notes attached to the item in Zotero (not annotations).
   Like annotations, only notes created or edited since the last
   import are added; earlier imports stay as they are.
   An edited note is added again under "Updated", the old version
   above it is kept, so delete it by hand if you don't need it. #}
{% if notes.length > 0 %}
## Zotero notes
{% endif %}
{% persist "zotero-notes" %}
{% if isFirstImport %}
{% set addedNotes = notes %}
{% set updatedNotes = [] %}
{% else %}
{% set changedNotes = notes | filterby("dateModified", "dateafter", lastImportDate) %}
{% set addedNotes = changedNotes | filterby("dateAdded", "dateafter", lastImportDate) %}
{% set updatedNotes = changedNotes | filterby("dateAdded", "dateonorbefore", lastImportDate) %}
{% endif %}
{% if addedNotes.length + updatedNotes.length > 0 %}
### Imported {{importDate | format("YYYY-MM-DD HH:mm")}}
{% for n in addedNotes %}
**New**
{{n.note}}

{% endfor %}
{% for n in updatedNotes %}
**Updated**
{{n.note}}

{% endfor %}
{% endif %}
{% endpersist %}

## Annotations
{# Only annotations made since the last import are added, so
   re-importing never duplicates highlights. #}
{% persist "annotations" %}
{% set newAnnotations = annotations | filterby("date", "dateafter", lastImportDate) %}
{% if newAnnotations.length > 0 %}
### Imported {{importDate | format("YYYY-MM-DD HH:mm")}}
{% for a in newAnnotations -%}
{% if a.annotatedText %}> {{a.annotatedText | replace("\n", " ")}}
{% endif -%}
{% if a.imageRelativePath %}> ![[{{a.imageRelativePath}}]]
{% endif -%}
> — {{a.colorCategory}} · [p. {{a.pageLabel}}](zotero://open-pdf/library/items/{{a.attachment.itemKey}}?page={{a.pageLabel}}&annotation={{a.id}})
{% if a.comment %}
**Note:** {{a.comment}}
{% endif %}
{% endfor %}
{% endif %}
{% endpersist %}
