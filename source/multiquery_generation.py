
from openai import OpenAI


client = OpenAI()


def generate_query_variations(input_query: str, num_variations: int = 3) -> list:
    """
    Generate multiple variations of a given query using the OpenAI API.

    Args:
        input_query (str): The original query to be varied.
        num_variations (int): The number of variations to generate.

    Returns:
        list: A list containing the generated query variations.
    """
    prompt = f"Generate {num_variations} variations of the following query:\n\n{input_query}\n\nVariations:"
    
    try:
        prompt = f"""You are an AI legal assistant. Your task is to generate 3 different versions 
        of the given user query to retrieve relevant documents from a vector database. 
        Provide these alternative questions separated by newlines, with no numbering or extra text.
        
        Original Query: {input_query}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            n=num_variations,
            stop=None,
            temperature=0.5,
        )
        
        variations = [choice.message.content.strip() for choice in response.choices]
        return variations
    
    except Exception as e:
        print(f"Error generating query variations: {e}")
        return []