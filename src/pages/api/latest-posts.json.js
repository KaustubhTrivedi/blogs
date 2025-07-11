import { getCollection } from 'astro:content';

export async function GET(context) {
	const posts = await getCollection('blog');
	
	// Sort posts by publication date (newest first)
	const sortedPosts = posts.sort((a, b) => new Date(b.data.pubDate) - new Date(a.data.pubDate));
	
	// Get the latest 10 posts
	const latestPosts = sortedPosts.slice(0, 10).map((post) => ({
		id: post.id,
		title: post.data.title,
		description: post.data.description,
		pubDate: post.data.pubDate.toISOString(),
		updatedDate: post.data.updatedDate ? post.data.updatedDate.toISOString() : null,
		url: `${context.site}/blog/${post.id}/`,
		heroImage: post.data.heroImage ? `${context.site}${post.data.heroImage.src}` : null,
		// Add a simple content preview (first 200 characters)
		contentPreview: post.body.substring(0, 200) + (post.body.length > 200 ? '...' : ''),
		wordCount: post.body.split(' ').length
	}));

	return new Response(JSON.stringify({
		success: true,
		totalPosts: posts.length,
		latestPosts: latestPosts,
		lastUpdated: new Date().toISOString()
	}, null, 2), {
		headers: {
			'Content-Type': 'application/json',
			'Cache-Control': 'public, max-age=300' // Cache for 5 minutes
		}
	});
} 