# 🐋Sentiment Analysis in Hotel Reviews of Buenaventura through Text Mining🐋

This project presents a sentiment analysis platform based on hotel reviews from **Buenaventura, Valle del Cauca, Colombia**. The goal is to analyze tourism perception using real review data and text mining techniques.

The platform allows visitors, researchers, and the local community to explore insights derived from hotel reviews, helping understand how travelers perceive hospitality services in the city.

Users can explore information such as:

- Sentiment distribution in hotel reviews
- Comparison between hotels in Buenaventura
- Overall visitor satisfaction trends
- Analysis of aspects frequently mentioned by guests

The project is publicly available at:

https://buenaventuradatos.com

This platform is freely accessible and aims to contribute to open, data-driven insights about tourism in Buenaventura.

## Tech Stack

- **Frontend:** Angular  
- **Data Processing:** Python / Text Mining techniques  
- **Database:** Supabase (PostgreSQL)  
- **Hosting:** Netlify  
- **Domain & DNS:** Cloudflare  
- **Data Visualization:** ECharts  

## Architecture (High Level)

The system follows a simple modern web architecture:

Data sources (hotel reviews) are collected and processed using text mining techniques.  
Processed data is stored in a database and exposed to the frontend application.  
The Angular web platform queries this processed data to generate visual insights and dashboards.

Infrastructure:

Cloudflare → Domain and DNS management  
Netlify → Hosting and deployment  
Supabase → Database and backend services  
Angular → Web application interface
