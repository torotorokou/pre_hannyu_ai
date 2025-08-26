def process_template_to_excel(template_key, dfs, csv_label_map, config):
    from utils.write_excel import write_values_to_template
    from logic.manage import template_processors

    processor_func = template_processors.get(template_key)
    if not processor_func:
        return None

    df = processor_func(dfs, csv_label_map)
    template_path = config["templates"][template_key]["template_excel_path"]
    return write_values_to_template(df, template_path)
