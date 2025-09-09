# 📈 Mutual Fund Performance Analysis Dashboard

A comprehensive Streamlit application for analyzing mutual fund performance data with advanced visualization and comparison tools.

## 🚀 Features

- **Performance Overview**: Comprehensive analysis across multiple time periods (1Y, 3Y, 5Y, 10Y)
- **Timeline Comparison**: Side-by-side comparison of Regular vs Direct plans
- **Consistency Analysis**: Track which funds consistently outperform their benchmarks
- **AUM Correlation**: Analyze relationship between fund size and performance
- **Risk-Return Analysis**: Evaluate funds based on risk categories
- **Performance Heatmap**: Visual representation of fund performance vs benchmarks
- **Investment Calculator**: Calculate final values for ₹1 Lakh investments

## 📊 Analysis Types

1. **Performance Overview**: Distribution charts and summary statistics
2. **Timeline Comparison**: Top performers across different time periods
3. **Consistency Analysis**: Funds that consistently beat benchmarks
4. **AUM Correlation**: Relationship between fund size and returns
5. **Risk-Return Analysis**: Performance by risk categories
6. **Performance vs Benchmark Heatmap**: Visual performance matrix
7. **Investment Calculator**: ₹1 Lakh investment analysis

## 🛠️ Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mutualfundanalyser.git
cd mutualfundanalyser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run mutualfundanalyserv2.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

## 📁 File Format

The application expects an Excel file with the following structure:
- Headers in row 5
- Fund data starting from row 6
- Required columns:
  - Scheme Name
  - Benchmark
  - Return columns for 1, 3, 5, and 10 years (Regular and Direct)
  - Daily AUM (Cr.)
  - Riskometer Scheme
  - NAV Date

## 🎨 Visualizations

- **Interactive Charts**: Plotly-powered visualizations with hover details
- **Dark Theme**: Optimized for better visual appeal
- **Color Coding**: 
  - 🟢 Green: Good performance (>0% vs benchmark)
  - 🟠 Orange: Average performance (0-4% vs benchmark)
  - 🔴 Red: Poor performance (≤0% vs benchmark)

## 📈 Key Metrics

- **Consistency Ratio**: Percentage of time periods a fund outperforms benchmark
- **Direct Advantage**: Additional returns from choosing direct plans
- **AUM Correlation**: Relationship between fund size and performance
- **Risk-Adjusted Returns**: Performance analysis by risk categories

## 🔧 Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **File Support**: Excel (.xlsx, .xls)

## 📝 Usage

1. Upload your mutual fund performance Excel file
2. Select analysis type from the sidebar
3. Explore interactive visualizations
4. Export results for further analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This tool is for educational and analysis purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## 🆘 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ for mutual fund analysis**
