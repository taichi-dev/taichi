import json
import os
from copy import deepcopy

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (CheckboxGroup, ColumnDataSource, DataTable, Div,
                          FactorRange, Select, TableColumn)
from bokeh.plotting import curdoc, figure, show
from bokeh.transform import factor_cmap

from misc import customize_palette, html_str_format, toolbox

file_dir = os.path.dirname(os.path.realpath(__file__))
results_file_path = os.path.join(file_dir, "results.json")
with open(results_file_path, 'r') as f:
    benchmark_results = json.load(f)

cases_list = sorted(list(benchmark_results['microbenchmarks']['cuda'].keys()))
# items
item_dict = {}
for case in cases_list:
    item_dict[case] = benchmark_results['microbenchmarks']['cuda'][case][
        'items']
# metrics
metric_dict = {}
for case in cases_list:
    metric_dict[case] = benchmark_results['microbenchmarks']['cuda'][case][
        'items']['metrics']


def filtering_data(origin_dict: dict, item_config=None):
    temp_dict = deepcopy(origin_dict)
    if item_config is None:
        return temp_dict
    case_list = list(temp_dict.keys())
    remove_list = []

    # item
    item_list = list(item_dict[case_selection.value].keys())
    for i in range(len(item_list)):
        tags = item_config.children[i].children[1]  #tgas_group: CheckboxGroup
        for i in range(len(tags.labels)):
            if i in tags.active:
                continue
            else:
                for case in case_list:
                    if (tags.labels[i] in temp_dict[case]['tags']) and (
                            case not in remove_list):
                        remove_list.append(case)
    #remove
    for case in remove_list:
        temp_dict.pop(case)

    return temp_dict


def create_items():
    item_list = list(item_dict[case_selection.value].keys())

    items_checkbox = []
    for item in item_list:
        tags = list(item_dict[case_selection.value][item])
        tgas_active = [num for num in range(len(tags))]
        if item == 'metrics':
            tgas_active = [0]
        tgas_group = CheckboxGroup(labels=tags,
                                   active=tgas_active,
                                   inline=True)
        tgas_group.on_change('active', update_figure)
        html_str_item = f"""
        <body>
            <p id="test">[ {item} ]:</p>
        </body>
        """
        row_x = row(Div(text=f'{html_str_format + html_str_item}'), tgas_group)
        items_checkbox.append(row_x)

    return column(items_checkbox, width=168)


def create_figure(grouped_by_item, filterd_data):
    # get active tags
    tags = list(item_dict[case_selection.value][grouped_by_item])
    active_tags = []
    for tag in tags:
        for data in filterd_data.values():
            if tag in data['tags']:
                active_tags.append(tag)
                break

    # nested data
    grouped_dict = {}
    grouped_name = []
    grouped_tag_name = []
    for tag in active_tags:
        for name, data in filterd_data.items():
            if tag in data['tags']:
                name_without_tag = name.replace(tag, '')
                if name_without_tag not in grouped_name:
                    grouped_name.append(name_without_tag)
                    grouped_dict[name_without_tag] = []
                grouped_tag_name.append((tag, name_without_tag))
                grouped_dict[name_without_tag].append(data['result'])

    grouped_result = []
    for i in range(len(active_tags)):
        for result_list in grouped_dict.values():
            if i < len(result_list):
                grouped_result.append(result_list[i])

    source = ColumnDataSource(data=dict(grouped_tag_name=grouped_tag_name,
                                        grouped_result=grouped_result))

    if len(grouped_tag_name) == 0:
        grouped_tag_name = [('null', 'null')]
        grouped_result = [0.0]

    y_axis_config = {'y_range': (0, max(grouped_result))}
    if 1 in config_checkbox.active:  # show_logaxis_y
        y_axis_config = {
            'y_axis_type': 'log',
            'y_range': (min(grouped_result) / 2, max(grouped_result))
        }

    p = figure(width=1440,
               height=720,
               title=case_selection.value,
               x_range=FactorRange(*grouped_tag_name),
               tooltips=[("case", "@grouped_tag_name"),
                         ("time_ms", "@grouped_result")],
               toolbar_location="above",
               tools=toolbox,
               **y_axis_config)

    # bar color
    color_agrs = {}
    if 2 in config_checkbox.active:  #show_colored_bar
        index_cmap = factor_cmap('grouped_tag_name',
                                 palette=customize_palette,
                                 factors=active_tags,
                                 end=1)
        color_agrs = {'fill_color': index_cmap}

    p.vbar(x='grouped_tag_name',
           top='grouped_result',
           bottom=min(grouped_result) / 2,
           width=1,
           source=source,
           line_color="white",
           **color_agrs)

    #default config
    p.y_range.start = min(grouped_result) / 2
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_text_color = 'white'
    p.xaxis.major_label_orientation = 0
    p.yaxis.axis_label = 'time: ms'
    # x label config
    if 0 in config_checkbox.active:  #show_label_x
        p.xaxis.major_label_text_color = 'black'
        p.xaxis.major_label_orientation = 1.55

    return p


def creat_table(filterd_data):
    names = [name for name in filterd_data.keys()]
    results = [value['result'] for value in filterd_data.values()]

    source = ColumnDataSource(data=dict(names=names, results=results))
    columns = [
        TableColumn(field="names", title="Case"),
        TableColumn(field="results", title="Result")
    ]
    data_table = DataTable(source=source,
                           columns=columns,
                           editable=True,
                           width=1440,
                           index_position=0,
                           index_header="row index",
                           index_width=60)

    return data_table


def create_group_by_item_selection():
    item_list = list(item_dict[case_selection.value].keys())
    grouped_by_item_selection = Select(title='grouped by item:',
                                       value=item_list[0],
                                       options=item_list)
    grouped_by_item_selection.on_change('value', update_figure)
    return grouped_by_item_selection


def create_config():
    config_txt = Div(text=f'config:')
    config_controls = column(config_txt, config_checkbox, width=168)
    return config_controls


def update_options(attr, old, new):
    layout.children[0].children[0].children[1] = create_config()
    layout.children[0].children[1].children[
        0] = create_group_by_item_selection()  # column2_item
    layout.children[0].children[1].children[1] = create_items()  # column2_item


def update_figure(attr, old, new):
    case_data = benchmark_results['microbenchmarks']['cuda'][
        case_selection.value]['results']
    filterd_data = filtering_data(case_data,
                                  layout.children[0].children[1].children[1])
    grouped_by_item = layout.children[0].children[1].children[0].value
    layout.children[1] = create_figure(grouped_by_item, filterd_data)
    layout.children[2] = creat_table(filterd_data)


def update(attr, old, new):
    update_options(attr, old, new)
    update_figure(attr, old, new)


# The box of case_selection does not change (global)
case_selection = Select(title='select case: ',
                        value=cases_list[0],
                        options=cases_list)
case_selection.on_change('value', update)
case_controls = column(case_selection, width=200)
# misc
config_checkbox = CheckboxGroup(
    labels=['show_label_x', 'show_logaxis_y', 'colored_bar'], active=[2])
config_checkbox.on_change('active', update_figure)
# group by item
item_list = list(item_dict[case_selection.value].keys())
grouped_by_item_selection = Select(title='grouped by item:',
                                   value=item_list[0],
                                   options=item_list)
grouped_by_item_selection.on_change('value', update_figure)
# set up layout
column1_case = column(case_controls, create_config(), width=256)
column2_item = column(grouped_by_item_selection, create_items())
row1_options = row(column1_case, column2_item)

case_data = benchmark_results['microbenchmarks']['cuda'][
    cases_list[0]]['results']
filterd_data = filtering_data(case_data, column2_item.children[1])
layout = column(row1_options,
                create_figure(grouped_by_item_selection.value, filterd_data),
                creat_table(filterd_data))

# if __name__ == '__main__':
# bokeh server
curdoc().title = "Microbenchmarks"
curdoc().add_root(layout)
