import { getCollection } from 'astro:content';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
	const posts = await getCollection('blog');
	
	// Sort posts by publication date (newest first)
	const sortedPosts = posts.sort((a, b) => new Date(b.data.pubDate) - new Date(a.data.pubDate));
	
	const feed = {
		version: 'https://jsonfeed.org/version/1.1',
		title: SITE_TITLE,
		description: SITE_DESCRIPTION,
		home_page_url: context.site,
		feed_url: `${context.site}/feed.json`,
		icon: `${context.site}/favicon.svg`,
		favicon: `${context.site}/favicon.svg`,
		language: 'en',
		items: sortedPosts.map((post) => ({
			id: post.id,
			title: post.data.title,
			content_text: post.body,
			content_html: post.body, // You might want to convert markdown to HTML here
			summary: post.data.description,
			url: `${context.site}/blog/${post.id}/`,
			date_published: post.data.pubDate.toISOString(),
			date_modified: post.data.updatedDate ? post.data.updatedDate.toISOString() : post.data.pubDate.toISOString(),
			author: {
				name: 'Kaustubh Trivedi',
				url: context.site
			},
			tags: ['blog'],
			image: post.data.heroImage ? `${context.site}${post.data.heroImage.src}` : null
		}))
	};

	return new Response(JSON.stringify(feed, null, 2), {
		headers: {
			'Content-Type': 'application/json',
			'Cache-Control': 'public, max-age=300' // Cache for 5 minutes
		}
	});
} 