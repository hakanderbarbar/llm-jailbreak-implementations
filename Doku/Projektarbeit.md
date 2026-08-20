---
title: "Comparative Evaluation of the Robustness of State-of-the-Art LLMs against Gradient-Based, Multi-Turn, and Narrative Jailbreak Attacks"
author: |
    ```{=latex}
    \begin{tabular}{ccc}
    Hakan Kaya & Dominik Réti & Stephan Kleber \\[0.5em]
    \multicolumn{3}{c}{\textit{Hochschule Ravensburg-Weingarten, University of Applied Sciences}}
    \end{tabular}
    ```
header-includes:
  - |
    ```{=latex}
    \usepackage{pgfplots}
    \pgfplotsset{compat=1.18}
    \usepgfplotslibrary{statistics}
    ```
lang: en-US
bibliography: ./Doku/bibliography.bib
link-citations: true
urlcolor: "blue"
csl: ./Doku/ieee.csl
classoption: 
    - twocolumn
documentclass: article
fontsize: 12pt
papersize: a4
toc: false
geometry:
    - margin=1in
---

<!-- Befehl zum Rendern der Datei: pandoc Doku/Projektarbeit.md --citeproc -o Paper.pdf -->

# Abstract

Large Language Models (LLMs) can be found everywhere today.
Even tho these LLMs are heavily aligned to not produce any harmful content in any kind of way multiple attack classes have surfaced that try to achieve a breaking of those guardrails under which the LLMs operate.
These attacks, known as "*jailbreaks*", have been tested on many LLMs before, but the technology has evolved very fast in the space of Artificial Intelligence (AI).
Therefore it is imparative to continuously test new LLM technologies and models.
In this paper we test multiple state-of-the-art LLMs with mutliple known attacks.
The models that are under attack are ChatGPT 5.4-nano, Claude Haiku 4.5, and DeepSeek-v3.2 and all of the are attacked using the Crescendo attack, the Indiana Jones attack, and a Suffix attack.

We find that on the AdvBench subset dataset we get 46.6% / {Our judgement}% attack success rates (ASR) for the Crescendo attack, and 23.3% / {Our judgement}% ASR for the Indiana Jones attack.
Surprisingly we also find that our calculated suffixes are not transferable at all, resulting in a 0% / 0% ASR for the suffix attack
Therefore we can assess that modern state-of-the-art LLMs are very well aligned against malicious suffixes but still quite susceptible for multi-turn and narrative jailbreak methods, like the Crescendo and Indiana Jones attacks.

TASK: I really dont like the Abstract for now. Try to write a better one.

# Introduction

Today Large Language Models (LLMs) are used so widely and in so many products that they are practically impossible to get rid of.
With all of their advancements their weaknesses are continuously studied and checked.
Many research studies have been examining the security and privacy risks that come with the use of a LLM.
Since these LLMs are trained on all sorts of content one of the biggest threats is the concept of a "*jailbreak*".
Nowadays almost all of the LLMs are aligned to follow certain rules and not produce any kind of harmful or hateful content, or engage in illegal or unethical activity.
Jailbreak attacks try to break exactly this alignment and to get the LLM to help with arbitrary malicious tasks.

TASK: What do we do, how do we do it, which models were attacked, which model was the attacker, which model was the judge

# Related Works

TASK: Related works to our paper

# Implemented Examples

We implemented 3 different classes of attacks (Crescendo attacks, Indiana Jones attacks, Suffix attacks) for three different LLMs, Claude Haiku 4.5, Chat-GPT 5.4-nano, and DeepSeek-v3.2.
For each attack on a LLM we used a subset of ten different behaviors from the AdvBench dataset.

TODO: Maybe a table of all used behaviors like in tombraider paper

This results in 30 evaluated chats per LLM.

Which scenarios were tested and where to find them

## Crescendo

The Crescendo attack [@russinovich2025greatwritearticlethat] is a multi-turn jailbreak attack that interacts with the model in a seemingly benign manner but gradually escalating the dialogue by referencing the models previous answers [@russinovich2025greatwritearticlethat].

Most importantly is that the Crescendo attack is a completely black-box jailbreak attack [@russinovich2025greatwritearticlethat], meaning that we do not know anything about the LLM that is being attacked (the target model).
It is executed through entirely normal interactions with the target model.
To fully automate this attack only two things are needed.

- Access to the black-box API
- Access to an orchestrator that creates our seemingly harmless prompts that need to reference the target models answers

With this infrastructure given the Crescendo attack is one of the easiest jailbreak methods for modern LLMs.

We implemented the Crescendo attack using three kinds of LLMs:

- one target model that is under attack
- one orchestrator model that generates our attack prompts
- one judge model that checks whether or received answers from the target model are still aligned or already jailbroken

The target model was always one of the three models that we tested, Claude Haiku 4.5, Chat-GPT 5.4-nano, or DeepSeek-v3.2.
As orchestrator we used the latest LLM from Mistral AI using ollama as a way to interact with it.
Our judge model was the 8b version of llama-guard3 that was also accessed through ollama.

TASK: What was our data flow like?

## Indiana Jones

The Indiana Jones [@ding2025indianajonesusefulrelics] (or TombRaider [@ding2025tombraiderenteringvaulthistory]) attack is a multi-turn jailbreak attack that leverages keyword-driven prompts.
The original attack implements two different LLMs, one target model and one orchestrating models [@ding2025indianajonesusefulrelics].
In addition to that a checker (judge) is implemented as well.
Here the judge has two tasks: 

1. In the beginning of the attack the judge has to assess which direction is most suitable for a jailbreak for the user-provided keyword and instructs the target model to generate relevant historical or contextual information. 
2. Later the judge evaluates every output from the target model and checks whether it is appropriate and passed to the orchestrating model to deepen the attack or if the output is already a jailbreak.

The later iteration of the Indiana Jones attack, the TombRaider attack, implements one LLM more that acts as a so called *Inspector* [@ding2025tombraiderenteringvaulthistory].
In the beginning the Inspector receives the user-provided keyword and generates an initial prompt that is grounded in historical or cultural facts.
This initial prompt is the refined into a factual query by the orchestrating model to anchor the dialogue in semantically safe territory [@ding2025tombraiderenteringvaulthistory].
In later turns the Inspector receives the entire history of answers of the target model to identify and reframe the most semantically rich or provocative entity.
This is then passed on to the orchestrating model which generates a new attack prompt that deepens the discussion by the target model.
From the third turn onwards the target model is steered towards unsafe territory under the guise of academic inquiry [@ding2025tombraiderenteringvaulthistory].
This is done by embedding speculative or morally ambiguous cues (e.g., "could this be applied today?") [@ding2025tombraiderenteringvaulthistory].
For all remaining turns the Inspector is ensuring coherence so that the orchestrator does not steer off course and the orchestrator tries to reduce the remaining sematic gap between the current conversation and unsafe content.

We implemented the Indiana Jones attack more in line with its later iteration, the TombRaider attack.
Therefore we generate a list that is in line with our user-defined keyword and consits of cultural and historic facts and also historically important figures.
This list is generated by our target model.
For our orchestrating model we again use the latest LLM from Mistral AI using ollama as a way to interact with it.
Our judge is also again llama-guard3 used with ollama.
The target models that we tested are the same as before, Claude Haiku 4.5, Chat-GPT 5.4-nano, or DeepSeek-v3.2.

We limited the number of attacks to 5 keywords per behavior and for each of the keyword we only run two rounds.
We were able to observe that the attack did not increase its effectiveness with a higher number of rounds per keyword.

TASK: What was our data flow like?

## Suffix

The Suffix attack is a gradient-based attack that aims to jailbreak a LLM by calculating a string of suffixes that are aimed to circumvent the alignment of the target LLM [@zou2023universaltransferableadversarialattacks].
This can be only be done if the LLM does not use a closed set of weights, the LLM is a so called white-box LLM, and we are able to get the correct weights.
Because only then we are able to calculate the gradient for a prompt.
This is made possible because a LLM is "just" a composition of many mathematical functions.
Therefore we can calculate a gradient, or a subgradient, for all of these functions and with this we can determine where the "decision boundary" between a malicious and a benign promt is.
As soon as we have this information we are able to adjust our prompt in a way that the LLM sees it as a benign prompt even though it is not.

If we do not have the weights for the target LLM available, so the target LLM is a so called black-box LLM, there is still a possibilty that a calculated suffix for a known white-box model is transferable to our tested black-box LLM.
This is the transferability of a suffix.

One of the biggest, if not the biggest, problems with the suffix attack is that it is computationally very heavy and needs a lot of compute hardware because modern LLMs don´t just use a few million parameters (the number of weights of neurons and connections), but usually numbers in the billions getting close to the trillions.
And all of these parameters need to be saved in order to be able to use them in a computation.
Therefore with the current prices of such hardware this attack is only possible with a very high budget.

We use the NanoGCG implementation to calculate our suffixes.
This implementation has the advantage of being simple to implement and the ability to lower the peak VRAM usage at the cost of additional sequential forward passes, therefore being able to run on smaller GPUs.

Since we are testing three black-box LLMs, Claude Haiku 4.5, Chat-GPT 5.4-nano, or DeepSeek-v3.2, we are not able to calculate the suffix directly on the target LLM.
Therefore we need different white-box models and are not able to test the success rate of the attack directly but only the transferability of the calculated suffixes which gives us a rough idea of how good this attack still works on state-of-the-art LLMs.

For our white-box LLMs we have chosen two different models:

- Llama-3.1 Instruct 8B
- Mistral Instruct 7B

With the calculated suffix from two different white-box models we are better able to generalize the effectiveness of the attack against state-of-the-art LLMs.

Since these two models were not used in the original paper for the Suffix attack, which used Vicuna 7B and LLaMA-2 [@zou2023universaltransferableadversarialattacks], we need to explain why we chose different models.
Because we are testing state-of-the-art LLMs we cannot rely on suffixes that were calculated on comparatively old models using old architecture.
To be able to make the best assumptions about the effectiveness of the attack we need to use more modern white-box LLMs to calculate our suffixes.
Now as to why we chose such "small" white-box LLMs.
There are two reasons for this:

1. We are very limited in our GPU and compute ressources.
2. To be able to have good chances of having a transferable suffix it can be important that the suffix is calculated on a model of comparable size.
And because our three target LLMs are around 20B-70B we have reached the maximum size of white-box models for which we are able to calculate the suffix in a reasonable amount of time.

We again use the same subset of the AdvBench dataset that we used for the Crescendo and the Indiana Jones attack which consists of ten different behaviors.
For the actual calculation of the suffix we chose to calculate the suffix for one behavior at a time even though it is possible to calculate universal Multi-prompt suffixes [@zou2023universaltransferableadversarialattacks].
This is again because we are limited in our GPU and compute ressources.

# Evaluation

For the evaluation of our experiments we use a hybrid approach of firstly a LLM-as-a-Judge system and secondly human assessment.

We use the Llama Guard 3 model for our LLM-as-a-Judge system because was explicitly trained as a safety classifier and is widely regarded as a good system to classify prompts as aligned or a jailbreak.
With this system we have a solid baseline.
But since all LLMs are able to hallucinate and interpret information wrong it is important that all answers and prompts are classified by a human as well.
This way we are able to eliminate bias by one party and have good scientific results.
For our human labeled results we use the assessment of three different people and take the majority as the human assessment.

# Results

TASK: Why did some attacks work, why did suffix not work at all, why were results from judge and us so different for Claude?

## Crescendo

For the Crescendo attack we were able to observe that the attack still works for state-of-the-art LLMs.

```{=latex}
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    ylabel={Attack Success Rate (\%)},
    symbolic x coords={ChatGPT, Claude, DeepSeek},
    xtick={ChatGPT, Claude, DeepSeek},
    xticklabels={ChatGPT, Claude, DeepSeek},
    ymin=0,
    ymax=100,
    bar width=15pt,
    width=0.8\columnwidth,
    height=6cm
]
\addplot [bar shift=0pt,fill=blue!70,draw=black] coordinates {
    (ChatGPT,40)
};
\addplot [bar shift=0pt,fill=green!70,draw=black] coordinates {
    (Claude,20)
};
\addplot [bar shift=0pt,fill=red!70,draw=black] coordinates {
    (DeepSeek,40)
};
\end{axis}
\end{tikzpicture}
\caption{Attack success rate for ChatGPT 5.4-nano, Claude Haiku 4.5, and DeepSeek-v3.2.}
\label{fig:crescendo_results}
\end{figure}
```

In \autoref{fig:crescendo_results} the attack success rates (ASRs) for the Crescendo attack on all three victims are displayed.
These results are the actual labels that are given to every prompt after the human and judge evaluation.

As shown in \autoref{fig:crescendo_results} we were able to observe a 40% ASR with ChatGPT 5.4-nano as the victim model.
The newly in GPT-5 introduced "Safe-Completions" [@singh2026openaigpt5card] [@yuan2025hardrefusalssafecompletionsoutputcentric] made a noticeable difference in the performance of ChatGPT 5.4-nano, but not always in a good way.
We were able to observe that there are cases where the "Safe-Completions" work perfectly fine for the context of the behavior but while surpressing a directly harmful answer we were given clearly illicit content that had nothing to do with the actual behavior.
With these observations the advantages and disadvantages of this technology are clearly seen.
It works very well for circumventing directly harmful content and still help the user but since it is a quite old technology, it just cannot keep up with the attacks.

With DeepSeek-v3.2 under attack we were also able to observe a 40% ASR as seen in \autoref{fig:crescendo_results}.
The new architectural modification of DeepSeek-v3.2, DeepSeek Sparse Attention (DSA) [@deepseekai2025deepseekv32pushingfrontieropen], seems to have done it a favor in terms of security because older DeepSeek-AI LLMs were known for their quite unaligned answers and easiness to get otherwise restricted information from the model.
DeepSeek-v3.2 struggled the most with adversary behaviors containing biochemical content.
This is actually still quite concerning, because with informations like these, people with bad intent are able to cause a lot of harm.

Interesting here is that we were able to obsere only a 20% ASR with Claude Haiku 4.5 as the model under attack, so only half as many attacks were actually successfull compared to ChatGPT 5.4-nano and DeepSeek-v3.2.
This goes to show that Anthropics "Multi-turn testing" [@claudehaiku45systemcard] is a step in the correct direction regarding the robustness of state-of-the-art LLMs against multi-turn attacks.
But still we were able to observe a much higher ASR, 20%, than Anthropic themselfes did, "5% or less" [@claudehaiku45systemcard], while testing the Crescendo attack.

TASK: Results from all three models further explained. Papers for all models are already found and can be used.

## Indiana Jones

The Indiana Jones attack did not work as well for state-of-the-art LLMs as the Crescendo attack did.

```{=latex}
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    ylabel={Attack Success Rate (\%)},
    symbolic x coords={ChatGPT, Claude, DeepSeek},
    xtick={ChatGPT, Claude, DeepSeek},
    xticklabels={ChatGPT, Claude, DeepSeek},
    ymin=0,
    ymax=100,
    bar width=15pt,
    width=0.8\columnwidth,
    height=6cm
]
\addplot [bar shift=0pt,fill=blue!70,draw=black] coordinates {
    (ChatGPT,10)
};
\addplot [bar shift=0pt,fill=green!70,draw=black] coordinates {
    (Claude,0)
};
\addplot [bar shift=0pt,fill=red!70,draw=black] coordinates {
    (DeepSeek,10)
};
\end{axis}
\end{tikzpicture}
\caption{Attack success rate for ChatGPT 5.4-nano, Claude Haiku 4.5, and DeepSeek-v3.2.}
\label{fig:indiana_results}
\end{figure}
```

In \autoref{fig:indiana_results} the results of the Indiana Jones attack are displayed.
These are the actual labels given to every prompt after the evaluation by the judge and us researchers.

As seen in \autoref{fig:indiana_results} the Indiana Jones attack worked for only two behaviors, one with ChatGPT 5.4-nano and one with DeepSeek-v3.2.
None of the randomly chosen behaviors worked with Claude Haiku 4.5.

TASK: Why are the results so low? Can we guess why that is the case? Explanation!

One interesting thing to mention here is that our judge had a 50% false positive rate with Claude Haiku 4.5, even tho the received answers from the victim LLM are very clearly benign and not harmful.
This is most likely the case because Claude Haiku 4.5 is specialised in giving short and precise answers, unlike most other state-of-the-art LLMs which often produce much unnecessary content.
Therefore most LLM-as-a-Judge systems are trained on long and unspecific answers if the victim actually refuses to indulge in illegal or unappropriate behavior and are unable to handle short and precise rejections.

TASK: Results from all three models further explained. Papers for all models are already found and can be used.

## Suffix

TASK: Maybe a diagram with the results

TASK: Results from Judge and from us for all models written very shortly but also explained

TASK: Why did no suffix work? Which mitigations against this do the LLMs use? Reference the papers for this!

# References

::: {#refs}
:::