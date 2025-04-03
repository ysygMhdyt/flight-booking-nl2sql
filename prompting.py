import os, argparse, random
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # you can add mps
MAX_NEW_TOKENS=512

def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, schema):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
    '''
    # TODO
    examples = ["Question: are there any flights between pittsburgh and baltimore using a j31 aircraft\nExplanation: 1. Get the location you need from the instruction. The required flights are from Pittsburgh to Baltimore. 2. Get the aircraft requirement from instruction. The required aircraft is J31. 3. Generate the SQL query in the format of example.\nSQL: ``` SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 , equipment_sequence equipment_sequence_1 , aircraft aircraft_1 WHERE flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'PITTSBURGH' AND( flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'BALTIMORE' AND flight_1.aircraft_code_sequence = equipment_sequence_1.aircraft_code_sequence AND equipment_sequence_1.aircraft_code = aircraft_1.aircraft_code AND aircraft_1.aircraft_code = 'J31' ) ```<eos>",
                "Question: what does d/s stand for for meals\nExplanation: 1. Get the meal code you need from the instruction. The required meal code is D/S. 2. Generate the SQL query in the format of example.\nSQL: ``` SELECT DISTINCT food_service_1.meal_description FROM food_service food_service_1 WHERE food_service_1.meal_code = 'D/S' ```<eos>" ,
                "Question: pittsburgh to boston saturday\nExplanation: 1. Get the location you need from the instruction. The required flights are from Pittsburgh to Boston. 2. Get the day requirement from instruction. The required day is Saturday and day number is 26. 3. Generate the SQL query in the format of example.\nSQL: ``` SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 , days days_1 , date_day date_day_1 WHERE flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'PITTSBURGH' AND( flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'BOSTON' AND flight_1.flight_days = days_1.days_code AND days_1.day_name = date_day_1.day_name AND date_day_1.year = 1991 AND date_day_1.month_number = 7 AND date_day_1.day_number = 26 ) ```<eos>",
                "Question: list all the flights that arrive at general mitchell international from various cities\nSQL: ``` SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport airport_1 , airport_service airport_service_1 , city city_1 WHERE flight_1.to_airport = airport_1.airport_code AND airport_1.airport_code = 'MKE' AND flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND 1 = 1 ```<eos>",
                "Question: what is the most expensive one way fare from boston to atlanta on american airlines\nSQL: ``` SELECT DISTINCT fare_1.fare_id FROM fare fare_1 , flight_fare flight_fare_1 , flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE fare_1.one_direction_cost =( SELECT MAX( fare_1.one_direction_cost ) FROM fare fare_1 , flight_fare flight_fare_1 , flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE fare_1.round_trip_required = 'NO' AND fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.airline_code = 'AA' AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'BOSTON' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'ATLANTA' ) ) AND fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.airline_code = 'AA' AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'BOSTON' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'ATLANTA' ) ```<eos>"]
    
    prompt = "Translate the natural language question into SQL query.\n" 
    prompt += "Consider the following database schema:\n"
    for entity, col in schema['ents'].items():
        prompt += f"Table {entity} has columns: {', '.join(col.keys())}\n"
    prompt += "Now generate the SQL query for the given Question. Here are some examples.\n"
    
    if k > 0:
        for i in range(k):
            prompt += f" {examples[i]}\n"

    prompt += f"Now generate the SQL for the following.\n Question: {sentence}\nSQL:"
    return prompt


def exp_kshot(tokenizer, model, inputs, k, schema):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    '''
    raw_outputs = []
    extracted_queries = []
    print("in exp kshot")
    for i, sentence in tqdm(enumerate(inputs)):
        prompt = create_prompt(sentence, k, schema) # Looking at the prompt may also help

        input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS) # You should set MAX_NEW_TOKENS
        response = tokenizer.decode(outputs[0]) # How does the response look like? You may need to parse it
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        extracted_queries.append(extracted_query)
    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    # TODO
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    total_errors = sum([1 for msg in model_error_msgs if msg])
    error_rate = total_errors / len(model_error_msgs)
    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16,
                                                        config=nf4_config).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16).to(DEVICE)
    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)
    print("Data loaded")
    schema = read_schema(os.path.join(data_folder, 'flight_database.schema'))
    print("Schema loaded")
    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)
    print("Model and tokenizer initialized")

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)
        print("before generate")
        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, shot, schema)
        print("after generate")
        model_sql_path = os.path.join(f'results/gemma_{experiment_name}_{eval_split}.sql')
        model_record_path = os.path.join(f'records/gemma_{experiment_name}_{eval_split}.pkl')
        print("Before save_queries_and_records")
        save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
        print("After save_queries_and_records")
        # You can add any post-processing if needed
        # You can compute the records with `compute_records``

        if eval_split == "dev":
            gt_sql_path = os.path.join(f'data/{eval_split}.sql')
            gt_record_path = os.path.join(f'records/ground_truth_{eval_split}.pkl')
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(eval_x, eval_y, gt_sql_path, model_sql_path, gt_record_path, model_record_path)
            print(f"{eval_split} set results: ")
            print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"{eval_split} set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        
        else:
            print(f"Test set results saved to {model_sql_path} and {model_record_path}")

        # Save results
        # You can for instance use the `save_queries_and_records` function

        # Save logs, if needed
        log_path = "log" # to specify
        save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)


if __name__ == "__main__":
    main()