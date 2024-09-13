import os
import requests
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool= False):
    """
    This method is used to scrape the linked in profile of people
    """
    if mock:
        linkedin_profile_url= "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
        response = requests.get(
            linkedin_profile_url,
            timeout=10,
        )
    else:
        pass

    data= response.json()

    data = {
        k:v
        for k,v in data.items()
        if v not in ["",'',None,[]]
        and k not in ["people_also_viewed","certifications"]
    }
    return data


if __name__=="__main__":
    print(
         scrape_linkedin_profile(
             linkedin_profile_url="some_URL",
             mock= True
         )
    )



