import gradio as gr
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load your trained model
with open('XGB final model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def predict_claim(Age,Diabetes,BloodPressure,AnyTransplants,AnyChronicDiseases,Height,Weight,KnownAllergies,
       HistoryOfCancerInFamily,NumberOfMajorSurgeries):
    # Prepare the input data in the same format as your training data
    input_data = [[Age,Diabetes,BloodPressure,AnyTransplants,AnyChronicDiseases,Height,Weight,KnownAllergies,
       HistoryOfCancerInFamily,NumberOfMajorSurgeries]]
    
    # Make a prediction
    prediction = loaded_model.predict(input_data)
    
    return round(prediction[0],0)

# Create a Gradio interface with customized output label
iface = gr.Interface(fn=predict_claim, 
                     inputs=[gr.Number(label="Age of person"), 
                             gr.Dropdown(label="Does Person has diabetes",choices=[('No',0),('Yes',1)]),
                             gr.Dropdown(label="Does person has Blood Pressure Problem",choices=[('No',0),('Yes',1)]),
                             gr.Dropdown(label="Any transplant",choices=[('No',0),('Yes',1)]),
                             gr.Dropdown(label="Any Chronic disease",choices=[('No',0),('Yes',1)]), 
                             gr.Number(label="Height in cms"),
                             gr.Number(label="Weight in Kg"),
                             gr.Dropdown(label="Any known allergies",choices=[('No',0),('Yes',1)]),
                             gr.Dropdown(label="History of Cancer in family",choices=[('No',0),('Yes',1)]),
                             gr.Dropdown(label="No of major surguries",choices=[('0',0),('1',1),('2',2),('3+',3)])],
                     
                     
                     
                     outputs=gr.Textbox(label="Insurance Premium (PA)"),
                     title="ABC Medical Insurance",
                     description="Enter details to predict the medical Insurance Premium",
                     allow_flagging='never')  # Set flagging to 'never'

# Launch the interface
iface.launch(inline=False)