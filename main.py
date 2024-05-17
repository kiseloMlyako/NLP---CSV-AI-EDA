# Description: This is the main file that will be used to run the Streamlit app
# Streamlit is used for the front-end of the app
import streamlit as st
# The csv agent is used to interact with the dataset
from langchain_experimental.agents.agent_toolkits import create_csv_agent
# The OpenAI model is used to interact with the dataset
from langchain.llms import OpenAI
# The Ollama model is used to interact with the dataset localy
from langchain_community.llms import Ollama
# Load the environment variables
from dotenv import load_dotenv
# Used to store the chat history
from dataclasses import dataclass
# Used to specify the type of the message
from typing import Literal

def main():
  # Lets load the environment variables (OpenAI API key)
  load_dotenv()

  @dataclass
  class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

  # Add a custom css to the app for a better UX
  def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

  # Initialize the session state for the history and conversation
  def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        # Set the temprature to 0 to get deterministic results
        llm = OpenAI(temperature=0)
        # Or use the Ollama model
        # llm = Ollama(model="llama3")

        # Create a csv agent and give it the dataset and the llm, along with verbose=True to show the thinking process in the terminal
        st.session_state.conversation = create_csv_agent(llm, csv_ds, verbose=True)

  # Callback function for when the user submits a chat message
  def on_click_callback():
      # Take the user input
      human_prompt = st.session_state.human_prompt
      # Check that input is not empty
      if human_prompt is not None and human_prompt != '':
        # Run the conversation with the user input
        llm_response = st.session_state.conversation.run(
            human_prompt
        )
        # Append the chat history with the user input and the AI response
        st.session_state.history.append(
            Message("human", human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response)
        )
        # Clear the input field
        st.session_state.human_prompt = None

  # Load the custom css
  load_css()

  # Set the title and headers of the app
  st.title("AI supported dataset EDA")
  st.header('AI supported dataset EDA')
  st.subheader('Input your .CSV dataset file and get the analysis answers with the help of AI!')

  # Upload the dataset csv file
  csv_ds = st.file_uploader("Upload a dataset", type=["csv"])

  # If the dataset is uploaded successfully we wait for the user to input a question
  if csv_ds is not None:
    
    # Initialize the session state
    initialize_session_state()

    #Prompt placeholders
    chat_placeholder = st.container()
    prompt_placeholder = st.form('chat-form')

    # Chat history
    with chat_placeholder:
      # Loop through the chat history and display the chat bubbles
      for chat in st.session_state.history:
          # Create a div element with the chat message using the custom css
          div = f"""
  <div class="chat-row
      {'' if chat.origin == 'ai' else 'row-reverse'}">
      <div class="chat-bubble
      {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
          &#8203;{chat.message}
      </div>
  </div>
          """
          # Display the chat bubble
          st.markdown(div, unsafe_allow_html=True)

      # Add some space between the chat history and the input field
      for _ in range(3):
          st.markdown("")

    # Chat input
    with prompt_placeholder:
      # Display a markdown message to the user
      st.markdown('_Press **Enter** to submit_')
      # Create a form with a text input field for the user to input a question
      cols = st.columns((6, 1))
      # The user input is stored in the session state found by the key 'human_prompt'
      cols[0].text_input('Chat', value='', label_visibility='collapsed', key='human_prompt')
      # Create a submit button that will call the on_click_callback function when clicked
      cols[1].form_submit_button('Submit', type='primary', on_click=on_click_callback)

if __name__ == '__main__':
    main()
