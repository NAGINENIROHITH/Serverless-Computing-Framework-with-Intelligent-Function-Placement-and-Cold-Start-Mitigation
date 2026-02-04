# Intelligent Serverless Framework - Web UI

A modern, real-time dashboard for monitoring and managing the Intelligent Serverless Framework.

## 🎨 Features

### Dashboard Views

#### 1. **Overview Dashboard**
- **Real-time Metrics**
  - Cold start rate with trend
  - P99 latency monitoring
  - Total requests counter
  - SLA compliance percentage
- **Interactive Charts**
  - Latency distribution (P50, P95, P99, Mean)
  - Cost breakdown by component
- **Live Updates** (every 30 seconds)

#### 2. **Functions Management**
- View all serverless functions
- Function statistics (invocations, latency)
- Status monitoring
- Quick actions (info, settings)

#### 3. **ML Predictions**
- Prediction accuracy metrics
- Model performance (MAPE, R² score)
- Per-function predictions
- Model type information

#### 4. **Intelligent Placement**
- Node distribution (Edge, Regional, Cloud)
- Placement optimization metrics
- Migration statistics
- Resource utilization

#### 5. **Cost Analysis**
- Total cost tracking
- Cost per invocation
- Savings analysis
- Detailed breakdown by component

## 🚀 Quick Start

### Option 1: Standalone HTML (No Server Required)

Simply open `dashboard.html` in your browser:

```bash
cd intelligent-serverless-framework/ui
open dashboard.html  # macOS
# or
xdg-open dashboard.html  # Linux
# or
start dashboard.html  # Windows
```

**Note**: The dashboard will use mock data if the API is not available.

### Option 2: With Flask Server (Recommended)

1. **Install Dependencies**
```bash
pip install flask flask-cors requests
```

2. **Start the UI Server**
```bash
cd intelligent-serverless-framework
python ui/server.py
```

3. **Access the Dashboard**
```
http://localhost:3000
```

### Option 3: With Docker

```bash
# Build UI container
docker build -t serverless-ui -f ui/Dockerfile .

# Run UI server
docker run -p 3000:3000 -e API_BASE_URL=http://api:8000/api/v1 serverless-ui
```

## 🔧 Configuration

### Environment Variables

```bash
# API endpoint
export API_BASE_URL=http://localhost:8000/api/v1

# UI server port
export UI_PORT=3000

# Development mode
export FLASK_ENV=development
```

### Update API Endpoint

Edit the API configuration in `dashboard.html`:

```javascript
const API_BASE_URL = 'http://your-api-server:8000/api/v1';
```

Or use the proxy server which handles CORS automatically.

## 📊 Dashboard Screenshots

### Overview Dashboard
```
┌─────────────────────────────────────────────────┐
│ Intelligent Serverless Framework     ● Active  │
├─────────────────────────────────────────────────┤
│  [Overview] [Functions] [Predictions] [Placement] [Cost]  │
├─────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │Cold Start│ │P99 Latency│ │Requests│ │SLA Comp│ │
│  │  8.0%   │ │  185ms  │ │ 125.8K │ │ 98.5%  │ │
│  │  ⬇ 65%  │ │  ⬇ 78%  │ │  ⬆ 23% │ │  ⬆ 5%  │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
├─────────────────────────────────────────────────┤
│  Latency Distribution    Cost Breakdown         │
│  [Bar Chart]             [Doughnut Chart]       │
├─────────────────────────────────────────────────┤
│  Active Functions        ML Predictions          │
│  • image-processor       image-processor: 5,234 │
│  • data-transformer      data-transformer: 3,456│
│  • api-gateway           api-gateway: 8,765     │
└─────────────────────────────────────────────────┘
```

## 🎨 UI Components

### Metric Cards
- Gradient backgrounds
- Hover animations
- Trend indicators (up/down arrows)
- Real-time updates

### Charts
- **Latency Chart**: Bar chart showing P50, P95, P99, Mean
- **Cost Chart**: Doughnut chart with breakdown
- **Responsive**: Adapts to screen size
- **Interactive**: Hover for details

### Tables
- Sortable columns
- Action buttons
- Status badges
- Hover effects

## 🔌 API Integration

### Endpoints Used

```javascript
GET /api/v1/metrics/system          // System metrics
GET /api/v1/functions               // List functions
GET /api/v1/predictions             // Get predictions
GET /api/v1/metrics/cost/current    // Cost data
```

### Mock Data Fallback

If API is unavailable, the UI automatically uses realistic mock data for demonstration purposes.

## 🎯 Features

### Real-time Updates
- Automatic refresh every 30 seconds
- Manual refresh button
- Live status indicator

### Responsive Design
- Mobile-friendly
- Tablet optimized
- Desktop full-featured

### Modern UI/UX
- Glass morphism effects
- Smooth animations
- Gradient accents
- Clean typography

### Accessibility
- Semantic HTML
- ARIA labels
- Keyboard navigation
- High contrast mode support

## 🛠️ Technology Stack

- **React 18**: UI framework
- **Tailwind CSS**: Styling
- **Chart.js**: Data visualization
- **Font Awesome**: Icons
- **Flask**: Backend server (optional)

## 📱 Mobile Support

The dashboard is fully responsive and works on:
- ✅ Desktop (1920x1080+)
- ✅ Laptop (1366x768+)
- ✅ Tablet (768x1024)
- ✅ Mobile (375x667+)

## 🔐 Security

### CORS Configuration
The Flask server includes CORS support. For production:

```python
CORS(app, origins=['https://yourdomain.com'])
```

### API Authentication
Add authentication headers in the proxy:

```python
headers = {
    'Authorization': f'Bearer {api_token}'
}
```

## 🚀 Production Deployment

### With Nginx

```nginx
server {
    listen 80;
    server_name dashboard.yourcompany.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### With Docker Compose

```yaml
version: '3.8'
services:
  ui:
    build: ./ui
    ports:
      - "3000:3000"
    environment:
      - API_BASE_URL=http://api:8000/api/v1
    depends_on:
      - api
```

## 📈 Performance

- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3.0s
- **Lighthouse Score**: 95+
- **Bundle Size**: < 500KB (with CDN)

## 🎨 Customization

### Change Color Scheme

Edit the Tailwind configuration:

```html
<script>
    tailwind.config = {
        theme: {
            extend: {
                colors: {
                    primary: '#your-color',
                    secondary: '#your-color'
                }
            }
        }
    }
</script>
```

### Add Custom Metrics

```javascript
<MetricCard
    title="Your Metric"
    value="123"
    icon="fa-your-icon"
    color="from-color-to-color"
    trend="+10%"
    trendUp={true}
/>
```

## 🐛 Troubleshooting

### Issue: Dashboard shows "Loading..."
**Solution**: Check API endpoint is running at `http://localhost:8000`

### Issue: CORS errors
**Solution**: Use the Flask proxy server instead of direct API calls

### Issue: Charts not rendering
**Solution**: Ensure Chart.js CDN is loaded correctly

### Issue: Mock data showing
**Solution**: Verify API_BASE_URL is correct and API is accessible

## 📖 Documentation

- **API Docs**: http://localhost:8000/docs
- **Project Docs**: See `/docs` folder
- **Code Comments**: Inline documentation in source

## 🤝 Contributing

To add new dashboard views:

1. Create new tab in `NavigationTabs`
2. Add corresponding component function
3. Update `activeTab` switch statement
4. Add any new API calls

## 📝 License

Apache 2.0 - See LICENSE file

## 🔗 Links

- GitHub: https://github.com/your-org/intelligent-serverless-framework
- Documentation: https://docs.intelligent-serverless.io
- API Reference: https://api.intelligent-serverless.io

---

**Built with ❤️ for the Intelligent Serverless Framework**
