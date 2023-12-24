import streamlit as st
import pandas as pd
import math

o_column_names=['Drug','Event','a','b','c','d','ROR','Upper bound',"Lower bound","dme_present"]

def pocess(df,target,treshold):
    result=pd.DataFrame(columns=o_column_names)
    drug_counts = df['Product Name'].value_counts()
    drug_dict = drug_counts.to_dict()
    min_count = treshold  
    filtered_dict0 = {l: w for l, w in drug_dict.items() if w >= min_count}
    for drug in filtered_dict0:
        filtered_df  = df[df['Product Name'] == drug]
        b=int(filtered_df['Case Num'].count())
        symptom_counts = filtered_df['Event Preferred Term'].value_counts()
        symptom_dict = symptom_counts.to_dict()
        min_count = treshold  
        filtered_dict = {k: v for k, v in symptom_dict.items() if v >= min_count}
        for event_term in filtered_dict:
            a=filtered_dict[event_term]
            event_term_count = df['Event Preferred Term'].value_counts()[event_term]
            c=event_term_count-a
            d_filtered_df = df[(df['Product Name'] != target) & (df['Event Preferred Term'] != event_term)]
            d=count = len(d_filtered_df)
            ROR=0
            lower_bound=math.nan
            upper_bound=math.nan
            if(c==0 or b==0):
                ROR=math.nan
                print("denominator has a null value")
                
            else:
                ROR = (a*d) / (b*c)  # replace with your ROR calculation if different
                lower_bound = math.exp(math.log(ROR) - (1.96 * math.sqrt(1/a + 1/b + 1/c + 1/d)))
                upper_bound = math.exp(math.log(ROR) + (1.96 * math.sqrt(1/a + 1/b + 1/c + 1/d)))
            

            print(f"95% CI Lower Bound: {lower_bound}")
            print(f"95% CI Upper Bound: {upper_bound}")
            new_row = {'Drug':drug,'Event':event_term,'a':a,'b':b,'c':c,'d':d,'ROR':ROR,'Upper bound':upper_bound,'Lower bound':lower_bound}
            result=pd.concat([result,pd.DataFrame([new_row])],ignore_index=True)
    return result

import math
import pandas as pd
dme_pd=pd.read_csv("dme1.csv")
column_name = 'PTs'
dme=dme_pd[column_name].tolist()
def process(df, target, threshold):
    result = pd.DataFrame(columns=o_column_names)
    # Filter 'Event Preferred Term' first
    event_counts = df['Event Preferred Term'].value_counts()
    filtered_event_dict = {event: count for event, count in event_counts.items() if count >= threshold or event in dme}

    for event_term in filtered_event_dict:
        filtered_df = df[df['Event Preferred Term'] == event_term]
        drug_counts = filtered_df['Product Name'].value_counts()
        min_count = threshold
        filtered_drug_dict = {drug: count for drug, count in drug_counts.items() if count >= min_count}

        for drug in filtered_drug_dict:
            drug_specific_df = filtered_df[filtered_df['Product Name'] == drug]
            a = filtered_drug_dict[drug]
            b = len(drug_specific_df)
            c = event_counts[event_term] - a
            d_filtered_df = df[(df['Product Name'] != drug) & (df['Event Preferred Term'] != event_term)]
            d = len(d_filtered_df)
            ROR = math.nan
            lower_bound = math.nan
            upper_bound = math.nan
            dme_present=0
            if(event_term in dme):
                dme_present=1
            if c == 0 or b == 0:
                ROR = math.nan
                print("Denominator has a null value")
            else:
                ROR = (a * d) / (b * c)  # replace with your ROR calculation if different
                lower_bound = math.exp(math.log(ROR) - (1.96 * math.sqrt(1/a + 1/b + 1/c + 1/d)))
                upper_bound = math.exp(math.log(ROR) + (1.96 * math.sqrt(1/a + 1/b + 1/c + 1/d)))

            print(f"95% CI Lower Bound: {lower_bound}")
            print(f"95% CI Upper Bound: {upper_bound}")
            new_row = {'Drug': drug, 'Event': event_term, 'a': a, 'b': b, 'c': c, 'd': d, 'ROR': ROR, 'Upper bound': upper_bound, 'Lower bound': lower_bound,'dme_present':dme_present}
            result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)

    return result


def main():
    st.title('Atlas Statistical Analysis')
    #st.image('./bg.jpeg',use_column_width=True)
    c1,c2=st.columns(2)
    target=c1.text_input("Enter the target column name")
    treshold=c1.number_input(label="Enter the treshold Frequency value",step=1)
    ror_threshold = st.slider('Select a ROR threshold', min_value=0, max_value=10)
    upper_bound_threshold = st.slider('Select an Upper Bound threshold',  min_value=0, max_value=10)
    lower_bound_threshold = st.slider('Select a Lower Bound threshold',  min_value=0, max_value=10)
    uploaded_file = c2.file_uploader("Choose a CSV file", type="csv")
    dfr=pd.DataFrame(columns=o_column_names)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if c2.button('Show Data'):
            st.write(df)
        if st.button('Process'):
            dfr=process(df,target,treshold)
            st.write(dfr)
            #ror_threshold = st.slider('Select a ROR threshold', min_value=int(dfr['ROR'].min()), max_value=int(dfr['ROR'].max()))
            #upper_bound_threshold = st.slider('Select an Upper Bound threshold', min_value=int(dfr['Upper bound'].min()), max_value=int(dfr['Upper bound'].max()))
            #lower_bound_threshold = st.slider('Select a Lower Bound threshold', min_value=int(dfr['Lower bound'].min()), max_value=int(dfr['Lower bound'].max()))
            #if st.button('Filter'):
            filtered_df = dfr[(dfr['ROR'] >= ror_threshold) & (dfr['Upper bound'] >= upper_bound_threshold) & (dfr['Lower bound'] >= lower_bound_threshold)]
            st.write(filtered_df)
            #st.write(dfr)
           
            #st.write(filtered_df)
            if st.button('Download'):
                dfr.to_csv('output.csv',index=False)
                st.success('File saved!')
            
            
    
    

if __name__ == '__main__':
    main()
