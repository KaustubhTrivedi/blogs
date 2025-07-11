# N8N Integration Guide

Your blog now has multiple endpoints that n8n can use to monitor for new posts:

## Available Endpoints

### 1. RSS Feed (XML)
- **URL**: `https://your-domain.com/rss.xml`
- **Format**: RSS 2.0 XML
- **Best for**: Traditional RSS readers and workflows

### 2. JSON Feed
- **URL**: `https://your-domain.com/feed.json`
- **Format**: JSON Feed 1.1
- **Best for**: Modern workflows that prefer JSON

### 3. Latest Posts API
- **URL**: `https://your-domain.com/api/latest-posts.json`
- **Format**: Custom JSON
- **Best for**: Simple workflows that need just the latest posts

## N8N Workflow Setup

### Option 1: Using RSS Trigger (Recommended)

1. **Add RSS Trigger Node**
   - Node Type: RSS Trigger
   - URL: `https://your-domain.com/rss.xml`
   - Polling Interval: 5-15 minutes (depending on your posting frequency)

2. **Configure the trigger**
   - Enable "Only trigger on new items"
   - Set appropriate polling interval

3. **Process new posts**
   - Add nodes to handle the RSS item data
   - Available fields: `title`, `description`, `link`, `pubDate`, `content`, `author`, `category`

### Option 2: Using HTTP Request Node

1. **Add HTTP Request Node**
   - Method: GET
   - URL: `https://your-domain.com/api/latest-posts.json`
   - Set up a schedule trigger (every 5-15 minutes)

2. **Compare with previous state**
   - Use a "Set" node to store the last known post ID
   - Compare new posts with stored data
   - Only process truly new posts

### Option 3: Using JSON Feed

1. **Add HTTP Request Node**
   - Method: GET
   - URL: `https://your-domain.com/feed.json`
   - Parse as JSON

2. **Process items**
   - Loop through `items` array
   - Check `date_published` for new posts

## Example N8N Workflow

```json
{
  "nodes": [
    {
      "id": "rss-trigger",
      "type": "n8n-nodes-base.rssTrigger",
      "parameters": {
        "url": "https://your-domain.com/rss.xml",
        "options": {
          "onlyNewItems": true
        }
      }
    },
    {
      "id": "process-post",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "// Process new blog post\nconst post = $input.first().json;\n\n// Example: Send to Discord\nreturn {\n  json: {\n    title: post.title,\n    description: post.description,\n    url: post.link,\n    author: post.author,\n    published: post.pubDate\n  }\n};"
      }
    }
  ]
}
```

## Important Notes

1. **Update your domain**: Replace `your-domain.com` in `astro.config.mjs` with your actual domain
2. **Caching**: All feeds are cached for 5 minutes to improve performance
3. **Content**: The RSS feed includes the full post content in the `content` field
4. **Timestamps**: All dates are in ISO 8601 format for easy parsing

## Testing Your Feeds

You can test the feeds locally by running:
```bash
npm run dev
```

Then visit:
- `http://localhost:4321/rss.xml`
- `http://localhost:4321/feed.json`
- `http://localhost:4321/api/latest-posts.json`

## Troubleshooting

- **Feed not updating**: Check that your `pubDate` in blog posts is set correctly
- **N8N not detecting new posts**: Ensure the polling interval is appropriate
- **Content missing**: Verify that your blog posts have the required frontmatter fields

## Next Steps

1. Update the `site` URL in `astro.config.mjs` to your actual domain
2. Deploy your blog
3. Set up your n8n workflow using one of the endpoints above
4. Test by creating a new blog post 