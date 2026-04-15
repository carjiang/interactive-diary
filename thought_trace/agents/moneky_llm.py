from types import SimpleNamespace
from typing import List

hardcoded_prompts = [
    "Elizabeth",
    "Elizabeth stepped into the hallway.<action>Benjamin arrived at the hallway.<no action>The box has the persimmon.<no action>The box is positioned in the hallway.<no action>Elizabeth conveyed the persimmon to the treasure chest.<action>The treasure chest is stored in the hallway.<no action>Benjamin exited the hallway.<no action>Elizabeth walked out of the hallway.<action>Benjamin tiptoed into the office.<no action>Benjamin is annoyed by the turnip.<no action>",
    "Elizabeth saw the long, narrow hallway ahead of her, likely lined with doors or windows, and possibly illuminated by overhead lights. She may have noticed the texture of the walls, the flooring beneath her feet, and any sounds echoing in the space.",
    "Elizabeth could not see the target context because the description only states that she stepped into the hallway without indicating her line of sight or awareness of Benjamin or the box containing the persimmon.",
    "Elizabeth saw the box containing the persimmon positioned in the hallway as she stepped in. She then conveyed the persimmon from the box to the treasure chest.",
    "Elizabeth could not see the target context because the treasure chest is stored in the hallway, but there is no indication that she was aware of its location or presence when she conveyed the persimmon. Additionally, Benjamin's exit from the hallway does not provide any information about Elizabeth's perception of the treasure chest.",
    "Elizabeth saw the hallway as she walked out, noting the box containing the persimmon and the treasure chest where she had just conveyed the fruit. She also observed Benjamin exiting the hallway before her.",
    "Elizabeth could not see the target context because she had already exited the hallway before Benjamin tiptoed into the office. Therefore, she was not present to perceive Benjamin's actions or feelings regarding the turnip.",
    "1. Elizabeth believed that the hallway would lead her to an important destination or information she needed.  2. She intended to explore the area, curious about what lay beyond the doors or windows.  3. Elizabeth felt a sense of urgency, thinking that she needed to move quickly to avoid missing an opportunity.  4. She was apprehensive about what she might encounter, considering the hallway's potential to hold surprises or challenges.",
    "Given that Elizabeth believes the hallway will lead her to an important destination or information she needs, it is reasonable to conclude that she would take the action of stepping into the hallway. This action aligns with her belief and indicates her intent to explore further in pursuit of what she is looking for. \n\nThe description of the hallway suggests that it is a space she is actively engaging with, rather than merely observing. Therefore, the probability of her stepping into the hallway is very high.\n\nAnswer: (a) Very Likely (Around 90%)",
    "Given Elizabeth's intention to explore the area and her curiosity about what lies beyond the doors or windows, it is reasonable to conclude that her action of stepping into the hallway aligns with her thoughts. She is likely motivated to investigate further, which supports the probability of her taking this action. The description of the hallway suggests it is an inviting space for exploration, reinforcing her decision to enter.\n\nTherefore, the probability of Elizabeth stepping into the hallway is very high.\n\nAnswer: (a) Very Likely (Around 90%)",
    "Given Elizabeth's sense of urgency and her need to move quickly to avoid missing an opportunity, it is reasonable to conclude that her action of stepping into the hallway aligns with her thoughts. The hallway represents a potential path toward whatever opportunity she is trying to seize. \n\nSince she is actively moving forward rather than hesitating or observing, this indicates a strong likelihood that she is motivated to proceed. Therefore, the probability of her stepping into the hallway is very high.\n\nAnswer: (a) Very Likely (Around 90%)",
    "Given Elizabeth's apprehension about what she might encounter in the hallway, it is reasonable to conclude that her decision to step into the hallway indicates a willingness to face whatever surprises or challenges lie ahead. The act of stepping into the hallway suggests that she is actively engaging with her environment rather than hesitating or avoiding it. \n\nSince she is aware of the potential for surprises and challenges, her action of stepping into the hallway aligns with her thoughts, indicating a level of determination or curiosity. Therefore, it is very likely that she would proceed with this action.\n\nAnswer: (a) Very Likely (Around 90%)",
    "Based on the current context, Elizabeth likely believed that the box containing the persimmon was significant and that conveying the persimmon to the treasure chest was an important task. She may have thought that this action would lead to a reward or fulfill a goal she had in mind, reinforcing her sense of purpose in the hallway.",
    "Based on the current context, Elizabeth likely believed that the box in the hallway contained something valuable or interesting, specifically the persimmon. Upon seeing the box, she recognized its contents and felt compelled to take action by conveying the persimmon to the treasure chest. This suggests that she believed the persimmon was important enough to be stored in the treasure chest, possibly indicating her perception of its value or significance in her exploration or quest.",
    "Based on the current context, Elizabeth likely believed that the box containing the persimmon was important and needed to be moved to the treasure chest. She may have felt a sense of purpose in conveying the persimmon, possibly viewing it as a valuable item that needed to be secured. Additionally, her previous sense of urgency might have influenced her belief that acting quickly to transfer the persimmon was necessary to achieve her goal. Overall, Elizabeth believed that she was fulfilling a task that was significant in the context of her current situation.",
    "Based on the current context, Elizabeth likely believed that the box containing the persimmon was an important item to retrieve and place in the treasure chest. Upon seeing the box in the hallway, she may have felt a sense of purpose or determination to complete the task of conveying the persimmon. Additionally, she might have felt a sense of relief or satisfaction upon finding the box, as it aligns with her goal of securing the persimmon. Overall, her belief centered around the significance of the persimmon and the importance of her actions in relation to it.",
    "To evaluate the likelihood of Elizabeth conveying the persimmon to the treasure chest, we need to consider the context and her thoughts. \n\n1. **Context Analysis**: Elizabeth stepped into the hallway and saw the box containing the persimmon. This indicates that she is aware of the box and its contents, which is crucial for her next action.\n\n2. **Thoughts Analysis**: Elizabeth believes that conveying the persimmon to the treasure chest is significant and that it aligns with a goal she has. This sense of purpose suggests that she is motivated to take action regarding the persimmon.\n\n3. **Action Probability**: Given that she has seen the box and recognizes the importance of the persimmon, it is very likely that she would take the action of conveying it to the treasure chest. Her belief in the significance of this task further supports the likelihood of her action.\n\nConsidering these points, the probability of Elizabeth conveying the persimmon to the treasure chest is very high.\n\nAnswer: (a) Very Likely (Around 90%)",
    "To evaluate the likelihood of Elizabeth conveying the persimmon to the treasure chest, we need to consider the context and her thoughts. \n\n1. **Context Analysis**: Elizabeth stepped into the hallway and saw a box containing a persimmon. The action of conveying the persimmon suggests that she recognized its value and importance, which aligns with her belief that it was significant enough to be stored in the treasure chest.\n\n2. **Thoughts Analysis**: Elizabeth's thoughts indicate that she believed the persimmon was valuable and felt compelled to act on that belief. This suggests a strong motivation to take the persimmon and place it in the treasure chest.\n\nGiven these points, it is reasonable to conclude that Elizabeth is very likely to convey the persimmon to the treasure chest, as her actions are driven by her perception of its value and her intent to store it.\n\nTherefore, the probability of her taking the action described is:\n\nAnswer: (a) Very Likely (Around 90%)",
    "To evaluate the probability of Elizabeth conveying the persimmon to the treasure chest, we need to consider the context and her thoughts. \n\n1. **Context Analysis**: Elizabeth stepped into the hallway and saw the box containing the persimmon. This indicates that she is aware of the box's presence and its contents, which is crucial for her next action.\n\n2. **Thoughts Analysis**: Elizabeth believes that the box containing the persimmon is important to retrieve and place in the treasure chest. Her feelings of purpose and determination suggest that she is motivated to complete this task.\n\n3. **Action Probability**: Given that she has seen the box and recognizes its significance, it is very likely that she would take action to convey the persimmon to the treasure chest. The context supports her awareness and intention, aligning with her goal.\n\nConsidering these points, the probability of Elizabeth conveying the persimmon to the treasure chest is very high.\n\nFinal Answer: Answer: (a) Very Likely (Around 90%)",
    "To evaluate the likelihood of Elizabeth walking out of the hallway, we need to consider her previous actions and thoughts. \n\n1. **Contextual Awareness**: Elizabeth has just conveyed the persimmon to the treasure chest, which indicates she has completed a task. This suggests she is likely to leave the area after finishing her objective.\n\n2. **Observations**: As she walks out, she notes the box containing the persimmon and the treasure chest, indicating she is aware of her surroundings and the significance of her actions. This awareness reinforces her sense of accomplishment.\n\n3. **Interaction with Benjamin**: Seeing Benjamin exit the hallway may also influence her decision to leave, as it could signify the end of an interaction or a transition to another part of her day.\n\nGiven these points, it is reasonable to conclude that Elizabeth is likely to walk out of the hallway after completing her task, as it aligns with her sense of purpose and the context provided.\n\nTherefore, the probability of her walking out of the hallway is:\n\nAnswer: (a) Very Likely (Around 90%)",
    "To evaluate the likelihood of Elizabeth walking out of the hallway, we need to consider her current state and thoughts. \n\n1. **Contextual Awareness**: Elizabeth has just conveyed the persimmon to the treasure chest and is aware of both the box and the treasure chest. This indicates she has a clear understanding of her surroundings.\n\n2. **Sense of Accomplishment**: Elizabeth feels a sense of accomplishment for completing her task, which suggests she is ready to move on from the hallway.\n\n3. **Observation of Benjamin**: She also notes Benjamin exiting the hallway, which may reinforce her decision to leave, as it indicates that there are no immediate concerns or distractions.\n\n4. **Final Action**: Given that she has completed her task and is aware of her surroundings, it is reasonable to conclude that she would walk out of the hallway.\n\nConsidering these points, the action of Elizabeth walking out of the hallway is very likely, as she has no reason to linger and has successfully completed her objective.\n\nFinal answer: Answer: (a) Very Likely (Around 90%)",
    "To evaluate the likelihood of Elizabeth walking out of the hallway, we need to consider her actions and thoughts in the context provided. \n\n1. **Contextual Awareness**: Elizabeth has just conveyed the persimmon to the treasure chest, indicating she has completed a task. This suggests she is aware of her surroundings, including the box and the treasure chest.\n\n2. **Sense of Accomplishment**: Elizabeth feels a sense of accomplishment after completing her task, which would likely motivate her to leave the hallway with a positive mindset.\n\n3. **Observation of Benjamin**: Noting Benjamin's exit may also influence her decision to leave, as it indicates that others are moving on, which could prompt her to do the same.\n\n4. **Final Action**: The action of walking out of the hallway is a natural progression after completing a task and observing her surroundings, especially since she has noted the important elements in the hallway.\n\nGiven these points, it is reasonable to conclude that Elizabeth is likely to walk out of the hallway after her actions and observations. Therefore, the probability of her doing so is high.\n\nFinal Answer: Answer: (a) Very Likely (Around 90%) \n\nAnswer: Very Likely (Around 90%)",
    "Based on the current context, Elizabeth likely believed that her actions in the hallway were complete and that she had successfully conveyed the persimmon to the treasure chest. Since she had already exited the hallway, she was unaware of Benjamin's annoyance regarding the turnip and therefore did not have any thoughts or beliefs related to his feelings or actions in the office. Elizabeth may have felt a sense of closure or satisfaction from her own task, focusing on her own experience rather than any external events occurring after her exit.",
    "Based on the current context, Elizabeth likely believed that her actions in the hallway were complete and that she had successfully conveyed the persimmon to the treasure chest. Since she had already exited the hallway, she was unaware of Benjamin's annoyance regarding the turnip and his subsequent actions in the office. Therefore, her focus would have been on her own experience and the significance of her task rather than on Benjamin's feelings or actions. Elizabeth may have felt a sense of accomplishment and purpose in her own quest, believing that she had contributed positively to whatever goal she was pursuing.",
    "Based on the current context, Elizabeth likely believed that she had successfully completed her task and was no longer concerned with the events happening in the hallway or the office. Since she had already exited the hallway, she may have felt a sense of closure regarding her actions with the persimmon and the treasure chest. She might have assumed that everything was in order and that her contribution was complete. Additionally, since she was not aware of Benjamin's annoyance with the turnip, she likely felt free to focus on her own thoughts and actions without any concern for his feelings or the situation in the office. Overall, Elizabeth believed she had made progress and was satisfied with her role in the recent events.",
    "Based on the current context, Elizabeth likely believed that she had successfully completed her task of conveying the persimmon to the treasure chest and felt a sense of accomplishment. Since she had already exited the hallway, she was unaware of Benjamin's actions or feelings regarding the turnip. Therefore, her thoughts would not be influenced by Benjamin's annoyance. Instead, she may have been reflecting on her own experience in the hallway, feeling satisfied with her achievement and possibly considering what her next steps might be now that she had completed her task. Overall, her beliefs centered around her success and the environment she had just navigated, without any knowledge of Benjamin's current situation.",
    "Elizabeth believed that the hallway would lead her to an important destination or information she needed.",
    "Elizabeth believed that the box containing the persimmon was significant and that conveying the persimmon to the treasure chest was an important task. She likely perceived the persimmon as valuable and felt a sense of purpose in securing it. Her actions were driven by the belief that transferring the persimmon was necessary to achieve her goal, and she may have experienced a sense of urgency or determination to complete this task. Overall, she recognized the importance of the persimmon and her role in ensuring it was safely stored in the treasure chest.",
    "Elizabeth believed that her actions in the hallway were purposeful and significant, particularly in conveying the persimmon to the treasure chest. She likely felt a sense of accomplishment for successfully completing this task and recognized the importance of the items within the hallway. Additionally, seeing Benjamin exit the hallway may have prompted her to reflect on their interaction and consider him a companion in her exploration. Overall, she felt that her efforts contributed to a larger goal and that she was making progress in her objectives.",
    "Elizabeth believed that she had successfully completed her task of conveying the persimmon to the treasure chest and felt a sense of closure and satisfaction from her actions. Since she had already exited the hallway, she was unaware of Benjamin's annoyance regarding the turnip and did not have any thoughts or concerns related to his feelings or actions in the office. Instead, her focus was on her own experience and the significance of her contribution, leading her to feel accomplished and content with her role in the recent events.",
    "Elizabeth believed that she had successfully completed her task of moving the persimmon to the treasure chest in the hallway. Since she had already left the hallway before Benjamin entered the office and became annoyed by the turnip, she had no knowledge of his actions or feelings. Therefore, her belief remained focused on her own completed action, with no awareness of anything that happened afterward.",
    "Elizabeth believed that everything in the hallway was already taken care of—specifically, that she had successfully moved the persimmon into the treasure chest before leaving. Because she had already exited the hallway, she had no awareness of Benjamin entering the office or feeling annoyed about the turnip. So, from her perspective, there were no new developments; she likely assumed things were proceeding normally and remained focused only on her completed task.",
    "Elizabeth believed several things at once: That the hallway might lead her to something important or useful. That it was worth exploring to see what was behind the doors or windows. That she needed to move with some urgency so she wouldn’t miss an opportunity. That there could be unexpected or challenging things ahead, which made her feel a bit apprehensive.",
    "Elizabeth believed that the persimmon in the box was important and that it needed to be moved to the treasure chest. Across the predictions, her belief consistently centers on the idea that transferring the persimmon was a meaningful or purposeful task—possibly because she saw it as valuable, necessary for a goal, or part of a larger objective she was trying to complete.",
    "Elizabeth believed that she had successfully completed her task of moving the persimmon into the treasure chest and that her actions in the hallway were finished. She likely felt a sense of accomplishment and closure, focusing only on what she had done. Importantly, she did not have awareness of any missing context (like uncertainty about the chest’s location earlier or Benjamin’s later actions or feelings). Her belief was centered on her own completed action and immediate experience.",
    "Elizabeth believed that she had successfully completed her task of moving the persimmon from the box to the treasure chest in the hallway. From her perspective, this action was finished and there was nothing left to do there. Because she had already left the hallway, she had no awareness of anything that happened afterward—such as Benjamin entering the office or his reaction to the turnip. Her belief was therefore limited to her own completed action, and she likely felt a sense of completion or satisfaction, assuming everything was in order.",
]

class MonkeyLLM:
    """
    A human-in-the-loop agent that mimics LLM agents.
    Every call to interact() or batch_interact() pauses and asks
    the user to input the response manually.
    """

    def __init__(self, kwargs: dict = None):
        """
        Initialize the agent.
        kwargs: dictionary of options like {"model": "manual", "temperature": 0, "max_tokens": 1024}
        """
        self.args = SimpleNamespace(**(kwargs or {}))
        # Provide defaults if missing
        if not hasattr(self.args, "model"):
            self.args.model = "manual"
        if not hasattr(self.args, "temperature"):
            self.args.temperature = 0
        if not hasattr(self.args, "max_tokens"):
            self.args.max_tokens = 1024

        # use this as an ID for feeding in the same prompt responses every time instead of manually pasting
        # for testing only.
        self.prompt_counter = -1

    def interact(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None) -> str:
        """
        Mimics a single LLM call. Prints the prompt and waits for user input.
        Returns: str (the response)
        """
        print("\n=======================")
        print("LLM PROMPT (interact)")
        print("=======================")

        if history:
            print("History:")
            for idx, msg in enumerate(history):
                role = "user" if idx % 2 == 0 else "model"
                print(f"{role}: {msg}")
        print("Prompt:")
        if system_prompt:
            print(system_prompt)
        print()
        print(prompt)

        self.prompt_counter += 1
        if self.prompt_counter > len(hardcoded_prompts)-1:
            # Ask user to type response
            response = input("\nType the response to return:\n> ")
        else:
            response = hardcoded_prompts[self.prompt_counter]
            print(response)

        return response

    def batch_interact(
        self,
        prompts: List[str],
        temperature=None,
        max_tokens=None,
        system_prompts=None,
        histories: List[List[str]] = None
    ) -> List[str]:
        """
        Mimics a batch LLM call. Loops over prompts and collects manual responses.
        Returns: List[str] (responses)
        """
        responses = []
        if isinstance(system_prompts, list):
            for idx, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts)):
                hist = histories[idx] if histories else None
                resp = self.interact(prompt, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, history=hist)
                responses.append(resp)
        else:
            for idx, prompt in enumerate(prompts):
                hist = histories[idx] if histories else None
                resp = self.interact(prompt, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompts, history=hist)
                responses.append(resp)
        return responses