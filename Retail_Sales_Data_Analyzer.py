import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class RetailAnalyzer:
    """
    Encapsulates all retail sales analysis functionality.
    
    Methods
    -------
    load_data(file_path)        : Read CSV using Pandas + validate
    calculate_metrics()         : Compute key sales metrics (NumPy)
    filter_data(condition)      : Filter by category or date range
    display_summary()           : Print formatted analysis report
    visualize()                 : Plot Bar, Line, Heatmap in matrix
    """
    
    def __init__(self):
        self.df = None
        self.filter_df = None
        self.file_path = None
    
    
    #Loading Data    
    def load_data(self, file_path):
        """
        Reads the dataset using Pandas.
        Uses if-else and loops to validate format and handle
        missing values.
        """
        
        # Path check
        if not os.path.exists(file_path):
            print(f"File not found!! : {file_path}")
            return False
        
        # File format check
        if not file_path.lower().endswith(".csv"):
            print("Only .csv files are supported.")
            return False
        
        # Load dataset 
        self.df = pd.read_csv(file_path)
        
        # Column check
        print("\nChecking for missing values:")
        num_cols = ["Price", "Quantity Sold", "Total Sales"]  # BUG FIX: typo "Total Salses"
        for col in num_cols:
            count = self.df[col].isna().sum()
            print(f"  {col} : {count}")
            if count > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                print("All missing values filled with median value.")
                
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df["Month"] = self.df["Date"].dt.month  
        
        # Make duplicate for filtered operations
        self.filter_df = self.df.copy()
        self.file_path = file_path
        
        print(f"\nDataset Loaded : {len(self.df)} Rows | {len(self.df.columns)} Columns")
        print(f"File : {file_path}")
        return True
    
    
    #Calculate Metrics
    def calculate_metrics(self):
        """
        Calculates total sales, average sales, and most popular
        product using NumPy on array values.
        """
    
        if self.df is None:
            print("\nNo data loaded.")
            return
        
        sal_array = self.df["Total Sales"].values
        qry_array = self.df["Quantity Sold"].values
        
        total_sales   = np.sum(sal_array)
        avg_sales     = np.mean(sal_array)
        median_sales  = np.median(sal_array)
        std_dev       = np.std(sal_array)
        max_sale      = np.max(sal_array)
        min_sale      = np.min(sal_array)
        
        # Growth Rate
        monthly_sales = self.df.groupby("Month")["Total Sales"].sum()
        growth_rate   = np.diff(monthly_sales.values)
        avg_growth    = np.mean(growth_rate)
        
        # Sales Percentage per Category
        cat_ttl   = self.df.groupby("Category")["Total Sales"].sum()
        sales_pct = np.round((cat_ttl / total_sales) * 100, 2)
        
        # Most Popular Product
        product_qty = self.df.groupby("Product")["Quantity Sold"].sum()
        top_product = product_qty.idxmax()
        top_qty     = product_qty.max()
        
        print(f"\n{'='*30} Sales Metrics Summary {'='*30}")
        print(f"\nTotal Records: {len(self.df)}")
        print(f"Total Sales: Rs. {total_sales:>13,.2f}")
        print(f"Average Sale/Record: Rs. {avg_sales:>13,.2f}")
        print(f"Median Sale: Rs. {median_sales:>13,.2f}")
        print(f"Std Deviation: Rs. {std_dev:>13,.2f}")
        print(f"Maximum Single Sale: Rs. {max_sale:>13,.2f}")
        print(f"Minimum Single Sale: Rs. {min_sale:>13,.2f}")
        print(f"Avg Monthly Growth: Rs. {avg_growth:>13,.2f}")
        print(f"Most Popular Product: {top_product} ({int(top_qty)} units)")
        print("-" * 55)
        
        print("\n  Category-Wise Sales:")  # BUG FIX: was "\Category..." (invalid escape)
        for cat in cat_ttl.index:
            pct = sales_pct[cat]
            cat_total = cat_ttl[cat]
            print(f"  {cat:<18}  Rs. {cat_total:>12,.2f}  ({pct}%)")  
        print("-" * 55)
    
    
    #Filter Data
    def filter_data(self, condition):
        """
        Filters data based on user-defined criteria.
        Supports: product category  OR  date range.
        
        Parameters
        ----------
        condition : dict
            {"type": "category",   "value": "Electronics"}
            {"type": "date_range", "start": "2024-01-01",
                                    "end":   "2024-06-30"}
        """
        
        if self.df is None:
            print("No data loaded!!")
            return
        
        ctype = condition.get("type", "")
        
        # Filter by category
        if ctype == "category":
            value = condition.get("value", "").strip()
            aval  = list(self.df["Category"].unique())
            match = None
            
            for cat in aval:
                if cat.lower() == value.lower():
                    match = cat
                    break
                
            if match:
                self.filter_df = self.df[self.df["Category"] == match].reset_index(drop=True)
                print(f"\nFiltered by Category : {match} →  {len(self.filter_df)} records") 
                
            else:
                print(f"\n  Category '{value}' not found.")
                print(f"  Available : {aval}")
                self.filter_df = self.df.copy()
                
        # Filter by Date Range
        elif ctype == "date_range":
            start = pd.to_datetime(condition.get("start"))
            end   = pd.to_datetime(condition.get("end"))
            
            if start > end:
                print("\n  Start date must be before end date.")
                self.filter_df = self.df.copy()
            else:
                mask = (self.df["Date"] >= start) & (self.df["Date"] <= end)
                self.filter_df = self.df[mask].reset_index(drop=True)
                print(f"\nFiltered by Date Range : {start.date()} → {end.date()} | {len(self.filter_df)} records") 
                
        else:
            print("\nNo filter applied. Showing all records.")
            self.filter_df = self.df.copy()
    
    
    #Summary of Data        
    def display_summary(self):
        """Displays a summary report of the (filtered) analysis."""
        
        data = self.filter_df
        
        if data is None or data.empty:
            print("\nNo data available for summary.")
            return
        
        print(f"\n{'='*30} Filtered Data Summary Report {'='*30}")
        print(f"\nRecords in current view: {len(data)}")
        
        # Aggregate by Category
        print("\n  Sales by Category:")
        cat_grp = data.groupby("Category")["Total Sales"].sum().sort_values(ascending=False)
        
        for cat, val in cat_grp.items():
            print(f"{cat:<18}  Rs. {val:>12,.2f}")
            
        # Aggregate by Product
        print("\nTop 5 Products by Revenue:")
        top5 = data.groupby("Product")["Total Sales"].sum().nlargest(5)
        
        for rank, (prod, val) in enumerate(top5.items(), start=1):
            print(f"{rank}. {prod:<22}  Rs. {val:>10,.2f}")
        
        arr = data["Total Sales"].values
        print(f"\nFiltered Stats:")
        print(f"Total : Rs. {np.sum(arr):>12,.2f}")
        print(f"Average : Rs. {np.mean(arr):>12,.2f}")
        print(f"Std Dev : Rs. {np.std(arr):>12,.2f}")
        print("-" * 55)
        
    
    #Visualization
    def visualize(self):  
        """
        Generates 3 plots arranged in a 2x2 matrix grid:
        [0,0] Bar Chart  — Total Sales by Product Category
        [0,1] Line Graph — Monthly Sales Trend over time
        [1,0] Heatmap    — Correlation (Price vs Qty Sold vs Total Sales)
        [1,1] (empty — reserved for future use)
        """
        
        if self.df is None:
            print("\nNo data found.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Retail Sales Dashboard", fontsize=16, fontweight="bold", y=0.98)
        
        #PLOT 1 [0,0]: Bar Chart 
        ax1 = axes[0, 0]
        cat_sales = (
            self.df.groupby("Category")["Total Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        bars = sns.barplot(data=cat_sales, x="Category", y="Total Sales", palette="Set2", ax=ax1)
        ax1.set_title("Bar Chart : Total Sales by Category", fontweight="bold", pad=10)
        ax1.set_xlabel("Product Category")
        ax1.set_ylabel("Total Sales (Rs.)")
        ax1.tick_params(axis="x", rotation=15)

        for bar in ax1.patches:
            ax1.annotate(
                f"Rs.{bar.get_height()/1e5:.1f}L",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9
            )

        #PLOT 2 [0,1]: Line Graph
        ax2 = axes[0, 1]
        monthly = self.df.groupby("Month")["Total Sales"].sum().reset_index()
        ax2.plot(
            monthly["Month"], monthly["Total Sales"],
            marker="o", color="blue", linewidth=2.2,
            markersize=6, markerfacecolor="white", markeredgewidth=2
        )
        ax2.fill_between(monthly["Month"], monthly["Total Sales"], alpha=0.10, color="blue")

        for _, row in monthly.iterrows():
            ax2.annotate(
                f"{row['Total Sales']/1e3:.0f}K",
                xy=(row["Month"], row["Total Sales"]),
                xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=7, color="blue"
            )

        ax2.set_title("Line Graph : Monthly Sales Trend", fontweight="bold", pad=10)
        ax2.set_xlabel("Month (2024)")
        ax2.set_ylabel("Total Sales (Rs.)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.set_xticks(monthly["Month"])

        #PLOT 3 [1,0]: Heatmap
        ax3 = axes[1, 0]
        num_df = self.df[["Price", "Quantity Sold", "Total Sales"]].copy()
        corr_matrix = num_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax3)
        ax3.set_title("Heatmap : Correlation (Price, Qty, Total Sales)", fontweight="bold", pad=10)
        ax3.tick_params(axis="x", rotation=30)

        #PLOT 4 [1,1]: Reserved 
        ax4 = axes[1, 1]
        ax4.axis("off")
        ax4.text(0.5, 0.5, "Reserved for Future Use", ha="center", va="center", fontsize=12, color="gray", style="italic")

        plt.tight_layout()
        plt.savefig("retail_dashboard.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("\nDashboard saved as retail_dashboard.png")


#Workflow 
def main():
    print(f"\n{'='*30} RETAIL SALES DATA ANALYZER {'='*30}")
    
    analyzer = RetailAnalyzer()
    
    # Step 1: Load Data
    print("\nStep 1: Input Validation")
    analyzer.load_data("retail_sales.csv")
    
    # Step 2: Metrics
    print("\nStep 2: Analysis and Metrics")
    analyzer.calculate_metrics()
    
    # Step 3: Filter — show all
    print("\nStep 3: Data Filtering  →  No filter (showing all records)")
    analyzer.filter_data({"type": "none"})
    
    # Step 4: Summary
    print("\nStep 4: Summary Report")
    analyzer.display_summary()
    
    # Step 5: Visualize
    print("\nStep 5: Visualization")
    analyzer.visualize()
    
    print("\nRetail Sales Analysis complete.\n")


if __name__ == "__main__":
    main()