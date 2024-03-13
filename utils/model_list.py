llms = {
    "facebook": {
        "facebook/galactica-6.7b": {
            "llm": "Galactica-6.7B",
            "color": '#A5CAD2',
            "alpha": 0.3,
            "hatch": "/",
            "n_params": 7,
            "Initial_Release": "Nov\n2022",
        },
        "facebook/galactica-30b": {
            "llm": "Galactica-30B",
            "color": '#758EB7',
            "alpha": 0.5,
            "hatch": "/",
            "n_params": 30,
            "Initial_Release": "Nov\n2022",
        },
        "facebook/galactica-120b": {
            "llm": "Galactica-120B",
            "color": '#6F5F90',
            "alpha": 0.9,
            "hatch": "/",
            "n_params": 120,
            "Initial_Release": "Nov\n2022",
        },
    },
    "falcon": {
        "tiiuae/falcon-40b": {
            "llm": "Falcon-40B",
            "color": '#E1C0D8',
            "alpha": 0.5,
            "hatch": None,
            "n_params": 40,
            "Initial_Release": "May\n2023",
        },
        "tiiuae/falcon-40b-instruct": {
            "llm": "Falcon-40B (instruct)",
            "color": '#E1C0D8',
            "alpha": 0.5,
            "hatch": "/",
            "n_params": 40,
            "Initial_Release": "May\n2023",
        },
        "tiiuae/falcon-180B": {
            "llm": "Falcon-180B",
            "color": '#D2A9B0',
            "alpha": 0.9,
            "hatch": None,
            "n_params": 180,
            "Initial_Release": "May\n2023",
        },
        "tiiuae/falcon-180B-chat": {
            "llm": "Falcon-180B (chat)",
            "color": '#D2A9B0',
            "alpha": 0.9,
            "hatch": "/",
            "n_params": 180,
            "Initial_Release": "May\n2023",
        },
    },
    "llama": {
        "meta-llama/Llama-2-7b-hf": {
            "llm": "Llama-2-7B",
            "color": '#D1DCE2',
            "alpha": 0.3,
            "hatch": None,
            "n_params": 7,
            "Initial_Release": "July\n2023",
        },
        "meta-llama/Llama-2-7b-chat-hf": {
            "llm": "Llama-2-7B (chat)",
            "color": '#D1DCE2',
            "alpha": 0.3,
            "hatch": "/",
            "n_params": 7,
            "Initial_Release": "July\n2023",
        },
        "meta-llama/Llama-2-13b-hf": {
            "llm": "Llama-2-13B",
            "color": '#B3DDD1',
            "alpha": 0.5,
            "hatch": None,
            "n_params": 13,
            "Initial_Release": "July\n2023",
        },
        "meta-llama/Llama-2-13b-chat-hf": {
            "llm": "Llama-2-13B (chat)",
            "color": '#B3DDD1',
            "alpha": 0.5,
            "hatch": "/",
            "n_params": 13,
            "Initial_Release": "July\n2023",
        },
        "meta-llama/Llama-2-70b-hf": {
            "llm": "Llama-2-70B",
            "color": '#80BEAF',
            "alpha": 0.9,
            "hatch": None,
            "n_params": 70,
            "Initial_Release": "July\n2023",
        },
        "meta-llama/Llama-2-70b-chat-hf": {
            "llm": "Llama-2-70B (chat)",
            "color": '#80BEAF',
            "alpha": 0.9,
            "hatch": "/",
            "n_params": 70,
            "Initial_Release": "July\n2023",
        },
    },
    "mistralai": {
        "mistralai/Mistral-7B-v0.1": {
            "llm": "Mistral-7B",
            "color": '#FA9284',
            "alpha": 1,
            "hatch": None,
            "n_params": 7,
            "Initial_Release": "Sept\n2023",
        },
        "mistralai/Mistral-7B-Instruct-v0.1": {
            "llm": "Mistral-7B (instruct)",
            "color": '#FA9284',
            "alpha": 1,
            "hatch": "/",
            "n_params": 7,
            "Initial_Release": "Sept\n2023",
        },
    },
}


llama_finetunes = {
    "llama": {
        "meta-llama/Llama-2-7b-chat-hf": {
            "llm": "Llama-2-7B (chat)",
            "color": '#8B9FC5',
            "alpha": 0.3,
            "hatch": "",
            "n_params": 7
        },
        "finetune_llama2_chat_7b": {
            "llm": "Llama-2-7B (chat) finetuned",
            "color": '#5874DC',
            "alpha": 0.3,
            "hatch": "*",
            "n_params": 7
        }
    }
}