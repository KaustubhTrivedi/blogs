import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
	const posts = await getCollection('blog');
	return rss({
		title: SITE_TITLE,
		description: SITE_DESCRIPTION,
		site: context.site,
		items: posts.map((post) => ({
			...post.data,
			link: `/blog/${post.id}/`,
			// Add content for better n8n processing
			content: post.body,
			// Add custom fields for n8n
			customData: `
				<guid>${post.id}</guid>
				<category>blog</category>
				<author>Kaustubh Trivedi</author>
			`,
		})),
	});
}
