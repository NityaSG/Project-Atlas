import streamlit as st
import pandas as pd
import math

o_column_names=['Drug','Event','a','b','c','d','ROR','Upper Bound',"Lower Bound"]

def process(df,target,treshold):
    result=pd.DataFrame(columns=o_column_names)
    filtered_df = df[df['Product Name'] == target]
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
        new_row = {'Drug':target,'Event':event_term,'a':a,'b':b,'c':c,'d':d,'ROR':ROR,'Upper Bound':upper_bound,'Lower Bound':lower_bound}
        result=pd.concat(result,pd.DataFrame([new_row]),ignore_index=True)
    return result



def main():
    st.title('Atlas Statistical Analysis')
    #st.image('./bg.jpeg',use_column_width=True)
    c1,c2=st.columns(2)
    target=c1.text_input("Enter the target column name")
    treshold=c1.number_input("Enter the treshold value")
    uploaded_file = c2.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if c2.button('Show Data'):
            st.write(df)
        if st.button('Process'):
            dfr=process(df,target,treshold)
            st.write(dfr)
            if st.button('Download'):
                dfr.to_csv('output.csv',index=False)
                st.success('File saved!')
            
    
    

if __name__ == '__main__':
    main()
