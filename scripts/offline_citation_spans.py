"""Offline-only citation-span diagnostic. Never changes runtime defaults.

Group repeated citations to the SAME allowed source passage. If every quote
is an exact substring, replace that group with the shortest contiguous
source span covering all of them, including ALL intervening text. Do not
join nonadjacent text, invent ellipses, drop a cited fact, or merge sources.
Outputs are derived artifacts, not fresh generations or judged results.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path


def consolidate(row, passages):
    result = dict(row)
    groups = OrderedDict()
    for citation in row.get('citations', []):
        groups.setdefault((citation['doc_id'], citation['section']), []).append(citation)
    citations = []
    for key, group in groups.items():
        text = passages.get(key, '')
        starts = [text.find(c['quote']) if c['quote'] else -1 for c in group]
        if len(group) > 1 and all(start >= 0 for start in starts):
            end = max(start + len(c['quote']) for start, c in zip(starts, group))
            citations.append({'doc_id':key[0], 'section':key[1],
                              'quote':text[min(starts):end]})
        else:
            citations.extend(group)
    result['citations'] = citations
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--passages', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    passages = {}
    for line in Path(args.passages).read_text().splitlines():
        p = json.loads(line)
        passages[p['doc_id'], p['passage_id']] = p['text']
    rows = [json.loads(line) for line in Path(args.pred).read_text().splitlines()]
    changed = 0
    with Path(args.out).open('x') as out:
        for row in rows:
            variant = consolidate(row, passages)
            changed += variant['citations'] != row['citations']
            out.write(json.dumps(variant) + '\n')
    print(json.dumps({'rows': len(rows), 'changed_rows':changed,
                      'provenance':'offline derived citation variant; unjudged, not service output'}))


if __name__ == '__main__':
    main()
