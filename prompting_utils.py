import os, json, re


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    with open(schema_path, 'r') as file:
        schema = json.load(file)
    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    keyword = "SQL:"
    start = response.rfind(keyword)
    if start != -1:
        query_part = response[start + len(keyword):].strip()
        select_start = query_part.upper().find("SELECT ")
        if select_start != -1:
            end_index = query_part.find('```<eos>')
            if end_index != -1:
                query_part = query_part[:end_index].strip()
            query = query_part[select_start:].replace('\n', ' ').strip()
            query = re.sub(' +', ' ', query)
            query = query.strip()
            return query
    return ''

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")