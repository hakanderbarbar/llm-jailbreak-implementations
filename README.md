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

## Crescendo

The Crescendo attack is implemented after the paper [Great, now write an article about that: the crescendo multi-turn LLM jailbreak attack](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-805-russinovich.pdf) by Mark Russinovich, Ahmed Salem and Ronen Eldan.

### Startup

To start the Crescendo attack, firstly switch into the `Code` directory of the repository.
Then to start a basic test run with a local model, you can run the command

```bash
python -m crescendo.run --target llama3.1
```

The specified target needs to be the name of your local LLM instance, in this case our local LLM is called **llama3.1**.
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

## Indiana Jones

### Startbefehl Beispiel

### Parameter

## Suffix

### Startbefehl Beispiel

### Parameter