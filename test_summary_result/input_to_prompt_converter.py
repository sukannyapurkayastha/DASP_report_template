# input_to_pompt_converter.py

###############################################################################
#                      PROMPT-BUILDING
###############################################################################

def build_prompt(input_data: str) -> str:
    return (
        "Instruction: Please summarize the following peer-review data \n\n"
        f"{input_data}\n\n"
        "Summary:\n"
    )

#######################################################
#                     LLaMA2 CODE
#######################################################

def build_llama2_prompt(input_data):
    # Define a clear, single instruction for the model
    default_instruction_p1 = """
I want you to summarize the following text.

## **Important Rules**:
- **Do not omit** any key points from the input data. Even minor details (like “typos” or “clarification requests”) must be included in the final summary.
- Keep formulars as they are.
- Maintain a **fluent** and **cohesive** text style.
- If multiple comments exist, **merge** them into a concise explanation, but keep all crucial info.


## **Reference Examples**

Input: "3.4 The method is claimed to generate SO(3)-invariant correspondences. However, in Tab. 1, even on the synthetic data, the I/SO(3) and SO(3)/SO(3) experiments perform unsimilarly (I would expect to have similar results per category, as it is on synthetic and clean data). Could this be explained?"
Output: "Summary: The reviewer questions why the method’s results differ significantly between I/SO(3) and SO(3)/SO(3) experiments if it is truly SO(3)-invariant, asking for an explanation despite the synthetic, clean data."}

-----------------------------------------------------------

Input: "The performance of the proposed solution is poor. For instance, Figure 2 most of the metrics overlap with existing solutions. The proposed algorithm seems to be only empirical and lacks theoretical guarantees and relies on MIIC as a base of the algorithm. To my understanding, the paper does not provide a guarantee that the algorithm always converges in a reasonable number of steps (even if not to a global minimum). Multivariate cross-information over a large set of variables may be very sensitive to noise, which I think is a weakness compared to algorithms that use cross-information only over small number of (e.g. triples) of variables."
Output: "Summary: The reviewer criticizes the algorithm’s performance, noting that results largely overlap with existing methods. They highlight a lack of theoretical convergence guarantees, heavy reliance on MIIC, and potential vulnerability to noise due to using multivariate cross-information on large variable sets."}

-----------------------------------------------------------

Input: "I also think that it will be beneficial to the reader if the connection between the theoretical results in Theorem 1 and the proposed algorithm are explained a bit more deeply with a more clear connection of how that theorem is being used."
Output: "Summary: The reviewer again urges the authors to clarify how Theorem 1 underpins the practical algorithm, highlighting the need for a clearer bridge between theory and implementation."}

-----------------------------------------------------------

Input: "Why do you pick the model subsets that you do in constructing each plot? For instance, plot 3b has a title \"Decision Tree: All Architectures\" but only plots GPT-2 and Mamba (what about Llama, and the other variations?) - Why apply Mamba only to Llama, and not to GPT-2? Since you mentioned compute was a bottleneck, why not keep either GPT-2 or Llama as a base model?"
Output: "Summary: The reviewer  emphasizes that this repeats earlier concerns about inconsistent model subset choices in plots, specifically referencing plot 3b’s label \"Decision Tree: All Architectures\" but only showing GPT-2 and Mamba. The commenter also questions the rationale behind applying Mamba only to Llama and not GPT-2, considering limited compute resources."}

-----------------------------------------------------------

Input: "1. The theme of this paper may not be closely related to the conference, as it is only an engineering specification and lacks theoretical explanation."
Output: "Summary: The reviewer feels the paper’s topic is tangential to the conference scope and lacks sufficient theoretical depth."}

-----------------------------------------------------------

Input: "4- Robust optimization has been extensively studied in the general optimization context. Many of such methods could be applicable to the RLHF/LLM problem and the method proposed in this paper is also also applicable to other settings. I do not see any comparisons to support this method is optimal for RLHF compared to existing robust optimization methods. 5- Related to the previous two comments, we we should have compared with other robust optimization methods and variations of applying the group loss 6- In Section 3, “The training of an AI assistant consists of three main stages...“ is not necessarily the case for all AI Assistants. I suggest revising this statement and connecting the presented method to more broader usecases as it is not really limited to this case."
Output: "Summary: The reviewer believes the paper should compare with robust optimization methods, clarifying that the described training pipeline might not be universal for all AI assistants. They want broader context and comparisons."}

-----------------------------------------------------------

Input: "Lines 60-61: What optimal train loss is being referred to here? There is no loss associated with ICL itself, right (as that only consists of in-context examples/demonstrations)? Does this loss refer to the ground truth functions being learned or the baselines? Line 80: which models are being referred to here? There is no “real training” during ICL itself. Are the functions/regression targets being referred to here or the baselines? The terminology used in the paper should be clarified. Wrt important missing information: - \"We replicate the function classes Linear Regression, Sparse Linear Regression, 2-Layer MLP Regression, and Decision Tree Regression from Garg et al. [6] as they present a wide range of \"difficulty\" for sequence models. In addition, to capture the existence of some ICL ability, we also regress onto the two function classes examined in Park et al. [14]: parity function with induced sparsity (Sparse Parity) and parallel associative recall (Vector MQAR).\" <- - How training instances are produced per task? How many test samples are produced per task? If this follows Garg et al., then each model is trained from scratch on 40 samples per task. Can you please clarify and state these in the main text? \"To determine task-specific ICL ability, our sequence models regress onto the functions shown above [14].\" <- It would help to clearly state the paper trains the models \"from scratch\" to in-context learn, as in previous works."
Output: "Summary: The reviewer requests clarity on what “optimal train loss” means in the absence of typical training for ICL, as well as which models are referenced in Line 80. They emphasize clarifying how tasks, training instances, and test samples are produced (citing Garg et al.) and reiterate that the paper should explicitly mention that models are trained from scratch to learn in context."}

-----------------------------------------------------------

Input: "In addition, there is a disconnection between the motivation of the work and the framework/evaluation. The framework does not account for user personalization, which is an important aspect when considering the ad ecosystem. So I am wondering how the paper is planning to incorporate user personalization in this framework and how Generative AI models can assist in this, especially when considering the privacy concerns that may arise from sharing user-specific data with companies that offer LLM solutions (e.g., OpenAI)."
Output: "Summary: The reviewer highlights a gap in addressing user personalization and potential privacy concerns when using LLM-based solutions for personalized ads."}

-----------------------------------------------------------

Input: "The reason why other methods are much better than LSTNet under the setting of I/I should be clarified. 4. For the SO(3)-equivariant and -invariant methods, some works for point cloud registration [2, 3, 4, 5] should also be discussed."
Output: "Summary: The reviewer wants an explanation for LSTNet’s underperformance in the I/I setting. They also suggest citing related SO(3)-equivariant/invariant point cloud registration work for completeness."}

-----------------------------------------------------------

# **Now**:

1. **Read** the input that follows.
2. **Do not** add extra commentary or disclaimers. **Only** produce one final summary in the required format, nothing else.
3. Start your summary with "Summary: The reviewers ..." or "Summary: The reviewer ..."
4. Write in sentences.

----

## **Input**: 
    """

    default_instruction_p2 = """


----

## **Output**:
    """

    full_prompt = default_instruction_p1 + input_data + default_instruction_p2
    return full_prompt