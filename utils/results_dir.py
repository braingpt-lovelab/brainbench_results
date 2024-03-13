def return_results_dir(use_human_abstract, llm):
    use_human_abstract = True
    if use_human_abstract:
        type_of_abstract = 'human_abstracts'
        human_abstracts_fpath = "data/human_abstracts.csv"
    else:
        type_of_abstract = 'llm_abstracts'
        llm_abstracts_fpath = "data/llm_abstracts_gpt-4.csv"
    results_dir = f"results/multiple_choice_harness/{llm.replace('/', '--')}/{type_of_abstract}"