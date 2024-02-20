from argparse import ArgumentParser
from dotenv import load_dotenv
from openai import OpenAI
import json, os, re
from tqdm import tqdm
import pandas as pd
from instructor import OpenAISchema
from pydantic import BaseModel, Field, conint, field_validator,confloat
from typing import List

class ChecklistItem(BaseModel):
    number: conint(ge=1) = Field(..., description="The item number")
    text: str = Field(..., description="The text of the checklist item")

class Checklist(OpenAISchema):
    items: List[ChecklistItem] = Field(..., description="The checklist items")

    def to_json(self):
        return self.model_dump_json()

    def to_dict(self):
        return self.model_dump()

    def to_markdown(self):
        txt = ""
        for item in self.items:
            txt += f"{item.number}. {item.text}\n"
        return txt

class ChecklistResponseItem(BaseModel):
    item_number: conint(ge=1) = Field(..., description="The chekclist item number.")
    isChecked: bool = Field(..., description="Indicates whether the item is contemplated by the candidate summary. It must be true only if it is contemplated by the candidate summary.")

class ChecklistResponse(OpenAISchema):
    """The responses from the evaluation checklist."""
    items: List[ChecklistResponseItem] = Field(..., description="The checklist items.")

    def call(self):
        results = []
        for item in self.items:
            results.append({
                "item": item.item_number,
                "contemplated":item.isChecked,
                # "reason":item.reason,
            })
        return results


class Evaluate:
    def __init__(self, model, data_file) -> None:
        self.model = model
        self.data_file = data_file

        load_dotenv()
        
        OPENAI_KEY = os.getenv('OPENAI_API_KEY')

        self.client = OpenAI(api_key=OPENAI_KEY)

        self.generate_checklist_instruction = open('generate_checklist_instruction.md').read()
        self.evaluate_checklist_instruction = open('evaluate_checklist_instruction.md').read()

        self.load_data()
    
    def load_data(self):
        df = pd.read_json(self.data_file, lines=True)
        self.data = df.to_dict(orient='records')

    def call_model(self,messages, functions):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            functions=functions,
            function_call={"name": functions[0]["name"]}
        )
        return response

    def generate_checklist(self,text):
        messages = [
            {"role":"system","content":self.generate_checklist_instruction},
            {"role":"user","content":f"##Original text\n\n{text}"}
        ]

        response = self.call_model(messages,functions=[Checklist.openai_schema])
        try:
            return Checklist.from_response(response)
        except Exception as e:
            messages.append(response.choices[0].message)
            messages.append({"role":"function","content":str(e),"name":"Checklist"})
            response = self.call_model(messages,functions=[Checklist.openai_schema])
            return Checklist.from_response(response)
            
    
    def evaluate_checklist(self, checklist, text, source_text):
        messages = [
            {"role":"system","content":self.evaluate_checklist_instruction.format(checklist=checklist.to_markdown())},
            # {"role":"user","content":f"## Source text\n\n{source_text}\n\n##Candidate text\n\n{text}"}
            {"role":"user","content":f"##Candidate text\n\n{text}"}
        ]

        response = self.call_model(messages,functions=[ChecklistResponse.openai_schema])
        return ChecklistResponse.from_response(response)

    def score_checklist(self, results):
        """Calculate the proportion of "yes" """
        return sum([1 if r["contemplated"] else 0 for r in results])/len(results)

    def replace_references(self, candidate, references):
        def replace_placeholder(match):
            index = int(match.group(1))  # Extract the index from the placeholder
            return references[index]["bibref"]  # Return the corresponding reference

        # Regular expression pattern to match 'REF{i}'
        pattern = r'REF(\d+)'

        # Replace all occurrences of the pattern in 'candidate'
        return re.sub(pattern, replace_placeholder, candidate)

    def evaluate(self):
        k_model = "gpt-4-1106-preview"
        output_file = f"v4_evaluated_{k_model}_{self.model}_{self.data_file}"
        lines = 0
        if not os.path.exists(output_file):
            open(output_file, 'w').close()
        else:
            # count lines
            with open(output_file, 'r') as f:
                lines = len(f.readlines())
        
        for item in tqdm(self.data[lines:]):
            text = item['section_text_in_survey']
            
            checklist = self.generate_checklist(text)
            item['checklist'] = checklist.to_dict()

            
            
            candidate = item["generated_section_text"][k_model]["text"]
            references = item["generated_section_text"][k_model]["references_sent_to_gpt"]

            candidate = self.replace_references(candidate, references)

            # print(checklist.to_markdown())
            # break
            
            evaluation = self.evaluate_checklist(checklist, candidate, text)
            item['evaluation'] = evaluation.call()
            item['score_checkeval'] = self.score_checklist(item['evaluation'])

            print(item['score_checkeval'])

            with open(output_file, 'a') as f:
                f.write(json.dumps(item)+'\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-16k')
    parser.add_argument('--data_file', type=str, default='scores_autosurvey.jsonl')
    args = parser.parse_args()

    evaluate = Evaluate(args.model, args.data_file)
    evaluate.evaluate()

