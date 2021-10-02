import os

import yaml
from taichi.core import settings


def extract_doc(doc_filename=None):
    statements_fn = os.path.join(os.path.dirname(__file__),
                                 '../taichi/ir/statements.h')
    with open(statements_fn, 'r') as f:
        statements = f.readlines()

    class_doc = {}

    for i in range(len(statements)):
        line = statements[i]
        start_pos = line.find('/**')
        if start_pos == -1:
            continue
        current_doc = line[start_pos + 3:].strip()
        doc_ends_at_line = 0
        for j in range(i + 1, len(statements)):
            next_line = statements[j]
            end_pos = next_line.find('*/')
            if end_pos != -1:
                doc_ends_at_line = j
                break
            next_line = next_line.strip()
            if next_line.startswith('*'):
                next_line = next_line[1:].strip()
            if next_line == '':  # an empty line
                current_doc += '\n'
            else:
                current_doc += ' ' + next_line
        current_doc = current_doc.strip()

        line = statements[doc_ends_at_line + 1]
        start_pos = line.find('class')
        if start_pos == -1:
            print('We only support doc for classes now. '
                  f'The following doc at line {i}-{doc_ends_at_line} '
                  'cannot be recognized:\n'
                  f'{current_doc}')
            continue
        class_name = line[start_pos + 5:].strip().split()[0]
        class_doc[class_name] = current_doc

    if doc_filename is None:
        doc_filename = 'ir_design_doc.yml'
    with open(doc_filename, 'w') as f:
        yaml.dump(class_doc, f, Dumper=yaml.SafeDumper)


def yml_to_md(yml_filename=None, md_filename=None):
    if yml_filename is None:
        yml_filename = 'ir_design_doc.yml'
    if md_filename is None:
        md_filename = 'ir_design_doc.md'
    with open(yml_filename, 'r') as f:
        doc_yml = yaml.load(f, Loader=yaml.SafeLoader)
    doc_md = ''
    for (class_name, class_doc) in doc_yml.items():
        doc_md += f'### {class_name}\n'
        doc_md += class_doc
        doc_md += '\n\n'
    with open(md_filename, 'w') as f:
        f.write(doc_md)


if __name__ == '__main__':
    extract_doc()
    yml_to_md()
    print('Done!')
