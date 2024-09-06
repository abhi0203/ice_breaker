from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import os

information = """
Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] ⓘ; born 17 September 1950)[a] is an Indian politician serving as the current Prime Minister of India since 26 May 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right wing Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.[3]

Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at the age of eight. At the age of 18, he was married to Jashodaben Modi, whom he abandoned soon after, only publicly acknowledging her four decades later when legally required to do so. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he rose through the party hierarchy, becoming general secretary in 1998.[b] In 2001, Modi was appointed Chief Minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat riots,[c] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim.[12] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him.[d] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[e]

In the 2014 Indian general election, Modi led the BJP to a parliamentary majority, the first for a party since 1984. His administration increased direct foreign investment, and it reduced spending on healthcare, education, and social-welfare programmes. Modi began a high-profile sanitation campaign, controversially initiated a demonetisation of banknotes and introduced the Goods and Services Tax, and weakened or abolished environmental and labour laws. Modi's administration launched the 2019 Balakot airstrike against an alleged terrorist training camp in Pakistan. The airstrike failed,[15][16] but the action had nationalist appeal.[17] Modi's party won the 2019 general election which followed.[18] In its second term, his administration revoked the special status of Jammu and Kashmir,[19][20] and introduced the Citizenship Amendment Act, prompting widespread protests, and spurring the 2020 Delhi riots in which Muslims were brutalised and killed by Hindu mobs.[21][22][23] Three controversial farm laws led to sit-ins by farmers across the country, eventually causing their formal repeal. Modi oversaw India's response to the COVID-19 pandemic, during which, according to the World Health Organization's estimates, 4.7 million Indians died.[24][25] In the 2024 general election, Modi's party lost its majority in the lower house of Parliament and formed a government leading the National Democratic Alliance coalition.
"""


if __name__ == "__main__":
    load_dotenv()
    print("Welcome back Abhiram to Langchain world. Hope you will complete the course!!!")
    print(os.environ["COOL_API_KEY"])

    # Create a summary template
    summary_template = """
    Given the information {information} above a person return 2 things
    1. A short summary
    2. Something interesting about it
    """

    # Create a summary template object
    summary_template_object = PromptTemplate(
        input_variable=["information"], template=summary_template
    )

    # Create a chat model that encircles the original LLM model
    # openai_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    ollama_llm = ChatOllama(model="llama3")

    # Create a chain of the template and chat model
    chain = summary_template_object | ollama_llm | StrOutputParser()

    # Invoke the chain model
    response = chain.invoke(input={"information": information})

    print(response)
