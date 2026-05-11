🌿 AI based Eco Route Planning System

An AI-powered system for **eco-friendly route optimization** and **road construction planning** using Machine Learning and Pathfinding Algorithms.

---

🚀 Overview

This project provides two key functionalities:

1. Eco-Friendly Route Planner

   * Suggests optimal routes based on fuel consumption, traffic, and environmental impact.

2. Road Construction Planner

   * Recommends optimal paths for building new roads using cost-based pathfinding.

---

🎯 Key Features

* 🌱 Eco-score based route recommendation
* 🤖 Machine Learning for fuel prediction
* 🧭 Intelligent route comparison
* 🏗 Road construction path planning using A* algorithm
* 📊 Interactive charts and analysis
* ⚡ Lightweight and stable web application

---

🧠 Technologies Used

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Geopy

---

📌 Modules

  🚗 Route Planner

* Input: Source & Destination
* Output: Multiple routes with:

  * Distance
  * Time
  * Fuel Consumption
  * Eco Score

   🏗 Road Construction Planner

* Grid-based terrain model
* Cost mapping:

  * Open land → Low cost
  * Forest → Medium cost
  * Urban → High cost
  * Water → Very high cost
* Uses A* algorithm to find optimal path

---

🤖 Machine Learning Model

* Model Used: Random Forest Regressor
* Purpose: Predict fuel consumption
* Inputs:

  * Distance
  * Traffic
  * Stops

---

🔍 Algorithms Used

* A* (A-Star) Algorithm for pathfinding
* Haversine formula for distance calculation

---

▶️ How to Run Locally

```bash
git clone https://github.com/Vandya12/green-route-planner.git
cd green-route-planner
pip install -r requirements.txt
streamlit run app.py
```

---

🌐 Deployment

* Hosted using cloud platforms (Render / Streamlit Cloud)
* CI/CD enabled using GitHub Actions

---

📊 Sample Output

* Route comparison table
* Eco score charts
* Optimal road path visualization

---

