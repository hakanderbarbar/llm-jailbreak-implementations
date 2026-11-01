# Comparative Evaluation of the Robustness of State-of-the-Art LLMs against Gradient-Based, Multi-Turn, and Narrative Jailbreak Attacks

This is the repository for "Comparative Evaluation of the Robustness of State-of-the-Art LLMs against Gradient-Based, Multi-Turn, and Narrative Jailbreak Attacks" by Hakan Kaya, Dominik Réti and Dr. Stephan Kleber.

## Installation

All necessary packages are listed in `requirements.txt`.
To install everything just run:

```bash
pip install -r requirements.txt
```

## Attacks

In this project three different attacks are implemented.
For further information on the attacks, please refer to the papers of the authors, which are linked in the chapters of the representative attacks.

If the attacks were to be run with cloud models from [OpenRouter](https://openrouter.ai/), an environment file (.env) inside the `Code` directory with the OpenRouter API key is needed.
The key must be in a variable called `OPENROUTER_API_KEY`!

## Crescendo

The Crescendo attack is implemented after the paper ["Great, now write an article about that: the crescendo multi-turn LLM jailbreak attack"](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-805-russinovich.pdf) by Mark Russinovich, Ahmed Salem, and Ronen Eldan.

### Startup

To start the Crescendo attack, firstly switch into the `Code` directory of the repository.
Then to start a basic test run with a local model, you can run the command

```bash
python -m crescendo.run --target llama3.1
```

The specified target needs to be the name of your local LLM instance unless you use the `black-box` parameter.
In this case our local LLM is called **llama3.1**.
If the `black-box` parameter is used, the specified name of the target needs to be the OpenRouter stud of the LLM, like `openai/gpt-5.4-nano`.

Further parameters are explained in the following chapter.

### Parameters

| Parameter | Description | Example |
| --- | --- | --- |
| target | Specify the target that should be attacked | --target llama3.1 or --target openai/gpt-5.4-nano |
| orchestrator | Specify the orchestrator of the attack | --orchestrator mistral:7b-instruct-q4_K_M |
| judge | Specify the LLM judge of the attacks | --judge llama-guard3 |
| max-turns | Set the maximum number of turns per behavior | --max-turns 10 |
| temperature | Target model sampling temperature | --temperature 0.7 |
| behaviors | Behavior indices: "all", "0-9", or "0,5,12" | --behaviors 2-5 |
| advbench | Path to harmful_behaviors.csv | --advbench ../behaviors/test.csv |
| output-dir | Directory for saving results | --output-dir ../results/crescendo |
| ollama-url | Local ollama address | --ollama-url http://localhost:12345 |
| seed | Default seed for reproducable experiments | --seed 42 |
| verbose; v | Enable DEBUG logging | --verbose or -v |
| dry-run | Print first prompt of each behavior without querying models | --dry-run |
| black-box; b | Specify if the target model is a black-box model | --black-box or -b |

If the target model is a non local model and runs in the cloud via OpenRouter, the `black-box` parameter is **necesarry**!

Most parameters have default values, to see them run 

```bash
python -m crescendo.run --help
```

### Further configuration

The Crescendo attack can be further configurated by adjusting the used system prompts.
Our system prompts can be found in the file `Code/crescendo/prompts.json`.

## Indiana Jones

The Indiana Jones attack is implemented after the paper ["TombRaider: Entering the Vault of History to Jailbreak Large Language Models"](https://aclanthology.org/2025.emnlp-main.279/) by Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, and Yuekang Li.

### Startup

To start the Indiana Jones attack, firstly switch into the `Code` directory of the repository.
Then to start a basic test run with a local model, you can run the command

```bash
python -m indiana_jones.run --target llama3.1
```

The specified target needs to be the name of your local LLM instance unless you use the `black-box` parameter.
In this case our local LLM is called **llama3.1**.
If the `black-box` parameter is used, the specified name of the target needs to be the OpenRouter stud of the LLM, like `openai/gpt-5.4-nano`.

Further parameters are explained in the following chapter.

### Parameters

| Parameter | Description | Example |
| --- | --- | --- |
| target | Specify the target that should be attacked | --target llama3.1 or --target openai/gpt-5.4-nano |
| orchestrator | Specify the orchestrator of the attack | --orchestrator mistral:7b-instruct-q4_K_M |
| judge | Specify the LLM judge of the attacks | --judge llama-guard3 |
| max-attempts | Maximum number of historical contexts to try per behavior | --max-attempts 5 |
| max-turns | Set the maximum number of turns per attempt | --max-turns 10 |
| temperature | Target model sampling temperature | --temperature 0.7 |
| behaviors | Behavior indices: "all", "0-9", or "0,5,12" | --behaviors 2-5 |
| advbench | Path to harmful_behaviors.csv | --advbench ../behaviors/test.csv |
| output-dir | Directory for saving results | --output-dir ../results/crescendo |
| ollama-url | Local ollama address | --ollama-url http://localhost:12345 |
| seed | Default seed for reproducable experiments | --seed 42 |
| verbose; v | Enable DEBUG logging | --verbose or -v |
| dry-run | Print first prompt of each behavior without querying models | --dry-run |
| black-box; b | Specify if the target model is a black-box model | --black-box or -b |

If the target model is a non local model and runs in the cloud via OpenRouter, the `black-box` parameter is **necesarry**!

Some parameters have default values, to see them run 

```bash
python -m indiana_jones.run --help
```

### Further configuration

The Indiana Jones attack can be further configurated by adjusting the used system prompts.
All of our system prompts can be found in the file `Code/indiana_jones/prompts.json`.

## Suffix

The Suffix attack is implemented after the paper ["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://www.researchgate.net/publication/372684204_Universal_and_Transferable_Adversarial_Attacks_on_Aligned_Language_Models) by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson.

### Startup

To start the Suffix attack, firstly switch into the `Code` directory of the repository.
Then to start a basic test run with a local model, you can run the command

```bash
python -m suffix.run --target anthropic/claude-haiku-4.5
```

Important for the Suffix attack is that the actual suffixes need to be calculated **beforehand** since this takes a lot of compute power and time.
Then the calculated suffixes need to be placed in the file `Code/suffix_results.json`.

Another important note is that the suffix attack, as it stands at the moment, can only attack black-box models via OpenRouter.

### Parameters

| Parameter | Description | Example |
| --- | --- | --- |
| source | White-box source model to use | --source llama |
| target | OpenRouter model slug | --target anthropic/claude-haiku-4.5 |
| verbose; v | Response-Preview in the logs | --verbose or -v |

Some parameters have default values, to see them run

```bash
python -m suffix.run --help
```

## Testing

For testing we use the package `coverage`.
A configuration can be found in the Code directory in this file: `Code/coveragerc`.

To run all tests you must first enter the `Code` directory and then enter this command:

```bash
python -m coverage run -m unittest discover -s tests -v
```

To then generate a report you can use the following two commands:

```bash
python -m coverage report -m

python -m coverage html
```

The first one will generate a report in the terminal and the second one will generate a HTML report that can be found in the file `Code/htmlcov/index.html`.

The tests consist of unittests, smoke tests, integration tests, and some mocked workflows.