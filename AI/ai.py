import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Load JSON Datasets ---
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

health_data = load_json('health_data.json')
career_data = load_json('career_data.json')
relationship_data = load_json('relationship_data.json')
fitness_data = load_json('fitness_data.json')

all_data = {
    "health": health_data,
    "career": career_data,
    "relationship": relationship_data,
    "fitness": fitness_data
}

# --- Gemini API Integration ---
def query_gemini_api(prompt, model_name="gemini-2.0-flash"):
    """Generate response using Gemini-2.0-flash."""
    try:
        model = GenerativeModel(model_name)
        chat = model.start_chat()
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Could not generate a response."

def format_explanation(user_input, raw_explanation):
    explanation = f"🤖 AI Explanation for your query: '{user_input}'\n\n"
    if raw_explanation:
        explanation += raw_explanation.strip()
    else:
        explanation += "The AI could not generate an explanation."
    return explanation

def get_relevant_info_all(query, all_data):
    query_lower = query.lower()
    relevant_info = []

    for category, data in all_data.items():
        if category == "health":
            # Check for specific conditions
            for condition in data["medical_conditions"]:
                if condition["name"].lower() in query_lower:
                    relevant_info.append(f"Regarding {condition['name']} (Health):")
                    # Include risk factors
                    if "risk_factors" in condition and any(rf["factor"].lower() in query_lower for rf in condition["risk_factors"]):
                        relevant_info.append("\n**Risk Factors:**")
                        for rf in condition["risk_factors"]:
                            if rf["factor"].lower() in query_lower:
                                relevant_info.append(f"- {rf['factor']}: {rf['explanation']}")
                    # Include diagnostic criteria (simplified)
                    if "diagnostic_criteria" in condition and "diagnose" in query_lower:
                        tests = ", ".join(condition["diagnostic_criteria"].get("tests", []))
                        thresholds = ", ".join([f"{k} {v}" for k, v in condition["diagnostic_criteria"].get("diagnostic_thresholds", {}).items()])
                        if tests or thresholds:
                            relevant_info.append(f"\n**Diagnosis:** It is diagnosed using tests like {tests} and thresholds such as {thresholds}.")
                    # Include treatment pathways
                    if "treatment_pathways" in condition and "treat" in query_lower:
                        treatment_info = "\n**Treatment Pathways:**\n"
                        for stage in condition["treatment_pathways"]:
                            actions = ", ".join(stage.get("recommended_actions", []))
                            treatment_info += f"- {stage['stage']}: {actions}\n"
                        relevant_info.append(treatment_info)
                    break  # Assuming we want info about one condition at a time

            # General questions about risk factors
            if "risk factors for" in query_lower:
                for condition in data["medical_conditions"]:
                    if condition["name"].lower() in query_lower:
                        relevant_info.append(f"Risk factors for {condition['name']} (Health) include:")
                        for rf in condition.get("risk_factors", []):
                            relevant_info.append(f"- {rf['factor']}: {rf['explanation']}")
                        break

        elif category == "career":
            # Check for specific career opportunities
            for career in data["career_opportunities"]:
                if career["name"].lower() in query_lower:
                    relevant_info.append(f"Regarding {career['name']} (Career):")
                    # Include career pathways
                    if "career_pathways" in career and "pathways" in query_lower:
                        relevant_info.append("\n**Career Pathways:**")
                        for cp in career["career_pathways"]:
                            relevant_info.append(f"- {cp['factor']}: {cp['explanation']}")
                    # Include entry requirements
                    if "entry requirements" in query_lower or "requirements for" in query_lower:
                        if "career_progression_criteria" in career and "entry_requirements" in career["career_progression_criteria"]:
                            requirements = ", ".join(career["career_progression_criteria"]["entry_requirements"])
                            relevant_info.append(f"\n**Entry Requirements:** {requirements}")
                    # Include advancement pathways
                    if "advancement" in query_lower or "progression" in query_lower:
                        if "advancement_pathways" in career:
                            advancement_info = "\n**Advancement Pathways:**\n"
                            for stage in career["advancement_pathways"]:
                                actions = ", ".join(stage.get("recommended_actions", []))
                                advancement_info += f"- {stage['stage']}: {actions}\n"
                            relevant_info.append(advancement_info)
                    break

        elif category == "relationship":
            # Check for specific relationship types
            for relationship in data["relationship_types"]:
                if relationship["name"].lower() in query_lower:
                    relevant_info.append(f"Regarding {relationship['name']} (Relationship):")
                    # Include relationship factors
                    if "relationship_factors" in relationship and "factors" in query_lower:
                        relevant_info.append("\n**Relationship Factors:**")
                        for rf in relationship["relationship_factors"]:
                            relevant_info.append(f"- {rf['factor']}: {rf['explanation']}")
                    # Include development pathways
                    if "development" in query_lower or "building" in query_lower:
                        if "relationship_development_pathways" in relationship:
                            development_info = "\n**Development Pathways:**\n"
                            for stage in relationship["relationship_development_pathways"]:
                                actions = ", ".join(stage.get("recommended_actions", []))
                                development_info += f"- {stage['stage']}: {actions}\n"
                            relevant_info.append(development_info)
                    break

        elif category == "fitness":
            # Check for specific fitness questions
            if "fitness_data" in data:  # Check if "fitness_data" key exists
                for item in data["fitness_data"]:  # Iterate through the array
                    if item["question"].lower() in query_lower:
                        relevant_info.append(f"Regarding {item['question']} (Fitness):")
                        # Include explanation
                        if "explanation" in item:
                            relevant_info.append(f"\n**Explanation:** {item['explanation']}")
                        # Include additional details
                        if "additional_details" in item:
                            relevant_info.append("\n**Additional Details:**")
                            for key, value in item["additional_details"].items():
                                if isinstance(value, list):
                                    relevant_info.append(f"\n{key.capitalize()}:")
                                    for detail in value:
                                        if isinstance(detail, dict):
                                            for sub_key, sub_value in detail.items():
                                                relevant_info.append(f"- {sub_key.capitalize()}: {sub_value}")
                                        else:
                                            relevant_info.append(f"- {detail}")
                                else:
                                    relevant_info.append(f"\n{key.capitalize()}: {value}")
                        break

    return "\n".join(relevant_info)

def explain_response_all(user_input, all_data):
    relevant_data = get_relevant_info_all(user_input, all_data)
    if relevant_data:
        prompt = f"Explain the following question using the provided information, presented as a list with bullet points:\n\nQuestion: {user_input}\n\nRelevant Information:\n{relevant_data}\n\nExplanation:"
    else:
        prompt = f"Explain the following question using bullet points:\n\nQuestion: {user_input}\n\nExplanation:"

    raw_response = query_gemini_api(prompt, model_name="gemini-2.0-flash")
    explanation = format_explanation(user_input, raw_response)
    return explanation

def chat():
    print("\n🤖 AI Chatbot: Hello! Ask me anything about health, career, relationships, or fitness, and I'll explain my answers in detail. Type 'exit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("\n🤖 AI: Goodbye! Have a great day!\n")
            break
        response = explain_response_all(user_input, all_data)
        print("\n🤖 AI:", response, "\n")

if __name__ == "__main__":
    vertexai.init(project="multi-domain-chatbot", location="us-central1") # Replace us-central1 with your chosen location
    chat()