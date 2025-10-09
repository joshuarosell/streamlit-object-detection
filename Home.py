import streamlit as st

st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')

st.title("📦YOLO Object Detection App")
# st.caption('This web application demonstrates Object Detection')

# Content
st.markdown("""
### This app detects from images, videos, and webcam stream.
- Identifies selected grocery items.
- [Click here for App](/YOLO_for_image/)  

Grocery items:
1. coffee_nescafe
2. coffee_kopiko
3. Lucky-Me-Pancit-Canton
4. Coke-in-can
5. Alaska-Milk
6. Century-Tuna
7. VCut-Spicy-Barbeque
8. Selecta-Cornetto
9. Nestle-Yogurt
10. Femme-Bathroom-Tissue
11. Spam-Classic
12. JnJ-Potato-Chips
13. Nivea-Deodorant
14. UFC-Canned-Mushroom
15. Libbys-Vienna-Sausage-can
16. Stik-O
17. NissinCupNoodles
18. Dewberry-Strawberry
19. Smart-C
20. Pineapple-juice-can
21. Nestle-Chuckie
22. Delight-Probiotic-Drink
23. Summit-Drinking-Water
24. almond_milk
25. Piknik
26. Rambutan
27. HS-Shampoo
28. irish-spring-soap
29. c2_na_green
30. colgate_toothpaste
31. 555-sardines
32. pic-a-chips
33. double-black
34. NongshimCupNoodles
           
            """)
