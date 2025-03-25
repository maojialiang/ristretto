import requests
requests.packages.urllib3.disable_warnings()

from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp.vlm import encode_image_file_to_base64
from ..dataset import DATASET_TYPE, DATASET_MODALITY


class RistrettoWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str,
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 max_tokens: int = 512,
                 cot_prompt=False,
                 temperature=0.0,
                 **kwargs):

        self.model = model
        self.cot_prompt = cot_prompt
        self.fail_msg = 'Failed to obtain answer via API. '
        self.default_params = {
            'top_k': 1,
            'best_of': 1,
            'do_sample': False,
            'max_tokens': max_tokens
        }
        if key is None:
            key = os.environ.get('GLMV_API_KEY', None)
            key = "test"
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://open.bigmodel.cn/dev/howuse/introduction)'
        )
        self.key = key
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)
    
    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True
        
    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt
        
    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        if int(os.environ.get("MAX_NEW_TOKENS", 0)) != 0:
            kwargs_default["max_new_tokens"] = int(os.environ.get("MAX_NEW_TOKENS", 0))
        self.kwargs = kwargs_default

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse'], dataset):
                prompt = question
            elif listinstr(['LLaVABench'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']

        if self.cot_prompt and not listinstr(['LLaVABench'], dataset):
            cot_prompt_with_final_answer = (
                "Your task is to answer the question below. "
                "Give step by step reasoning before you answer, and when you're ready to answer, "
                "please use the format \"Final answer: ..\""
                "\n\n"
                "Question:"
                "\n\n"
                "{question}"
            )
            cot_prompt_wo_final_answer = (
                "Your task is to answer the question below. "
                "Give step by step reasoning. "
                "\n\n"
                "Question:"
                "\n\n"
                "{question}"
            )

            if listinstr(['MMVet'], dataset):
                cot_prompt = cot_prompt_wo_final_answer
            else:
                cot_prompt = cot_prompt_with_final_answer

            question_orig = line['question']
            if listinstr(['MathVerse', 'MathVision'], dataset):
                question_orig = question_orig.split('Question:', 1)[-1].strip()
                question_orig = question_orig.replace('Choices:\n', '').strip()

            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            if options_prompt.strip():
                question_orig = f'{question_orig}\n{options_prompt}'

            prompt = cot_prompt.format(question=question_orig)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
    
    def build_msgs(self, msgs_raw, dataset=None):
        msgs = cp.deepcopy(msgs_raw)
        content = []
        for i, msg in enumerate(msgs):
            if msg['type'] == 'text':
                content.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                content.append(dict(type='image_url', image_url=dict(url=encode_image_file_to_base64(msg['value']))))

        ret = [dict(role='user', content=content)]
        return ret

    def generate_inner(self, inputs, dataset=None, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        self.set_max_num(dataset)
        inputs = [inputs] if isinstance(inputs, str) else inputs

        messages = self.build_msgs(msgs_raw=inputs, dataset=dataset)

        url = "http://localhost:5000/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Request-Id': 'remote-test',
            'Authorization': f'Bearer {self.key}'
        }
        params = cp.deepcopy(self.default_params)

        if dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            params["upscale_flag"] = True
        params["max_num"] = self.max_num

        payload = {
            'model': self.model,
            'messages': messages,
            **params,
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
        # import ipdb
        # ipdb.set_trace()
        output = []
        try:
            assert response.status_code == 200
            for line in response.iter_lines():
                data = json.loads(line.decode('utf-8').lstrip('data: '))
                output.append(data['choices'][0]['message']['content'])
            answer = ''.join(output).replace('</s>', '')
            if self.verbose:
                self.logger.info(f'inputs: {inputs}\nanswer: {answer}')
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')
            return -1, self.fail_msg, ''

    def set_max_num(self, dataset):
        if int(os.environ.get("MAX_PATCH_NUM", 0)) != 0:
            max_patch_num = int(os.environ.get("MAX_PATCH_NUM", None))
            self.max_num = max_patch_num
            return None
            
        if dataset is None:
            self.max_num = 6
            return None
        # res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'MME-RealWorld', 'VCR_EN', 'VCR_ZH']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        else:
            self.max_num = 6


class RistrettoAPI(RistrettoWrapper):

    def generate(self, message, dataset=None):
        return super(RistrettoAPI, self).generate(message, dataset=dataset)
