# Data

This folder should contain the following files to run the training pipeline:

- `Sale_012025.csv` through `Sale_072025.csv` - monthly transaction records
- `Unit.csv` - unit conversion factors

**However these files are proprietary and not distributed with this repository.**

### Expected CSV Schema

**Sales files:**
| Column | Description |
|--------|-------------|
| `SaleId` | Unique transaction ID |
| `ProcessDate` | Transaction date (YYYY-MM-DD) |
| `ClientId` | Client identifier |
| `ItemId` | Product identifier |
| `Quantity` | Units sold |
| `Price` | Transaction value |
| `ItemUnitCode` | Unit of measurement |

**Unit file:**
| Column | Description |
|--------|-------------|
| `ItemId` | Product identifier |
| `Code` | Unit code |
| `ConvFact1` | Conversion factor 1 |
| `ConvFact2` | Conversion factor 2 |
